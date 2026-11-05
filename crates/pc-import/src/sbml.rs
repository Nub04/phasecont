//! SBML (levels 2 and 3, core only) to `.ode` text.
//!
//! The reader keeps what an ODE model needs: compartments, species, parameters, reactions with
//! MathML kinetic laws, rate rules and assignment rules. Each reaction becomes a fixed quantity
//! named by the reaction id, and each species that reactions change becomes a state whose
//! right side is the signed sum of those reaction rates (divided by the compartment size for a
//! concentration). Nothing is inlined, so the text is about as large as the XML.
//!
//! These are refused by name, with their position: events, function definitions, initial
//! assignments, algebraic rules, delays, stoichiometry math, conversion factors, rules that
//! change a compartment size or a stoichiometry, and packages marked `required`. A species
//! quantity, a parameter value or a compartment size that the model needs and the file does not
//! give is an error. Nothing defaults to 0.

use std::collections::{BTreeSet, HashMap, HashSet};

use roxmltree::{Document, Node as X, ParsingOptions};

use crate::emit::{Imported, OdeModel, State, emit};
use crate::error::{ImportError, Kind, Result, line_col};
use crate::expr::{Expr, Op};
use crate::mathml;

/// Largest SBML file the reader accepts.
pub const MAX_XML: usize = 16 << 20;
/// Most XML nodes the reader parses.
const NODES_LIMIT: u32 = 1_000_000;

struct Src<'t> {
    text: &'t str,
}

fn local<'a>(n: X<'a, '_>) -> &'a str {
    n.tag_name().name()
}

impl Src<'_> {
    fn err(&self, at: usize, kind: Kind, msg: impl Into<String>) -> ImportError {
        let (l, c) = line_col(self.text, at);
        ImportError::at(kind, msg, l, c)
    }

    fn id(&self, n: X) -> Result<String> {
        match n.attribute("id") {
            Some(s) if !s.trim().is_empty() => Ok(s.trim().to_string()),
            _ => Err(self.err(
                n.range().start,
                Kind::Syntax,
                format!("a `{}` element has no `id`", local(n)),
            )),
        }
    }

    fn req(&self, n: X, attr: &str) -> Result<String> {
        match n.attribute(attr) {
            Some(s) if !s.trim().is_empty() => Ok(s.trim().to_string()),
            _ => Err(self.err(
                n.range().start,
                Kind::Syntax,
                format!("a `{}` element has no `{attr}`", local(n)),
            )),
        }
    }

    fn num(&self, n: X, attr: &str) -> Result<Option<f64>> {
        let Some(raw) = n.attribute(attr) else {
            return Ok(None);
        };
        let v: f64 = raw.trim().parse().map_err(|_| {
            self.err(
                n.range().start,
                Kind::Syntax,
                format!("`{attr}=\"{}\"` is not a number", raw.trim()),
            )
        })?;
        if !v.is_finite() {
            return Err(self.err(
                n.range().start,
                Kind::Unsupported,
                format!(
                    "`{attr}=\"{}\"` is not finite, and `.ode` has no literal for it",
                    raw.trim()
                ),
            ));
        }
        Ok(Some(v))
    }

    fn flag(&self, n: X, attr: &str) -> Result<Option<bool>> {
        match n.attribute(attr).map(str::trim) {
            None => Ok(None),
            Some("true" | "1") => Ok(Some(true)),
            Some("false" | "0") => Ok(Some(false)),
            Some(other) => Err(self.err(
                n.range().start,
                Kind::Syntax,
                format!("`{attr}=\"{other}\"` is not true or false"),
            )),
        }
    }
}

struct Compartment {
    id: String,
    size: Option<f64>,
}

struct Species {
    id: String,
    compartment: String,
    amount: Option<f64>,
    conc: Option<f64>,
    only_substance: bool,
    boundary: bool,
    constant: Option<bool>,
    at: usize,
}

struct Param {
    id: String,
    value: Option<f64>,
    constant: Option<bool>,
    at: usize,
}

struct Local {
    name: String,
    value: Option<f64>,
    at: usize,
}

struct Reaction {
    id: String,
    /// Species and signed stoichiometry: negative for reactants, positive for products.
    terms: Vec<(String, f64)>,
    math: Expr,
    locals: Vec<Local>,
    at: usize,
}

struct RuleDef {
    target: String,
    rate: bool,
    math: Option<Expr>,
    at: usize,
}

/// A value-carrying name that may become a `par`.
struct Candidate {
    name: String,
    value: Option<f64>,
    what: &'static str,
    at: Option<usize>,
}

fn xml_error(e: &roxmltree::Error) -> ImportError {
    match e {
        roxmltree::Error::NodesLimitReached => ImportError::new(
            Kind::TooLarge,
            "the XML has more nodes than the importer reads",
        ),
        roxmltree::Error::DtdDetected => ImportError::new(
            Kind::Xml,
            "the file has a DOCTYPE, which the importer refuses",
        ),
        other => {
            let p = other.pos();
            ImportError::at(Kind::Xml, other.to_string(), p.row, p.col)
        }
    }
}

/// Translates an SBML file (UTF-8 bytes) to `.ode` text.
pub fn from_sbml(bytes: &[u8]) -> Result<Imported> {
    if bytes.len() > MAX_XML {
        return Err(ImportError::new(
            Kind::TooLarge,
            format!("the SBML file is longer than {MAX_XML} bytes"),
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|e| {
        ImportError::new(
            Kind::Xml,
            format!(
                "the file is not UTF-8 text (invalid byte after offset {})",
                e.valid_up_to()
            ),
        )
    })?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let opts = ParsingOptions {
        allow_dtd: false,
        nodes_limit: NODES_LIMIT,
        ..ParsingOptions::default()
    };
    let doc = Document::parse_with_options(text, opts).map_err(|e| xml_error(&e))?;
    let src = Src { text };
    Reader::new(&src).run(&doc)
}

struct Reader<'s, 't> {
    src: &'s Src<'t>,
    warnings: Vec<String>,
}

impl<'s, 't> Reader<'s, 't> {
    fn new(src: &'s Src<'t>) -> Self {
        Reader {
            src,
            warnings: Vec::new(),
        }
    }

    fn run(mut self, doc: &Document) -> Result<Imported> {
        let src = self.src;
        let root = doc.root_element();
        if local(root) != "sbml" {
            return Err(src.err(
                root.range().start,
                Kind::Syntax,
                format!("the root element is `{}`, not `sbml`", local(root)),
            ));
        }
        let level = src.req(root, "level")?;
        match level.as_str() {
            "2" | "3" => {}
            "1" => {
                return Err(src.err(
                    root.range().start,
                    Kind::Unsupported,
                    "SBML level 1 is not translated; convert the file to level 2 or 3",
                ));
            }
            other => {
                return Err(src.err(
                    root.range().start,
                    Kind::Unsupported,
                    format!("SBML level `{other}` is not translated"),
                ));
            }
        }
        let l3 = level == "3";
        for a in root.attributes() {
            if a.namespace().is_some() && a.name() == "required" && a.value().trim() == "true" {
                return Err(src.err(
                    root.range().start,
                    Kind::Unsupported,
                    "the file requires an SBML package, which the importer does not translate",
                ));
            }
        }
        let Some(model) = root
            .children()
            .find(|c| c.is_element() && local(*c) == "model")
        else {
            return Err(src.err(root.range().start, Kind::Syntax, "the file has no `model`"));
        };
        if model.attribute("conversionFactor").is_some() {
            return Err(src.err(
                model.range().start,
                Kind::Unsupported,
                "a model `conversionFactor` is not translated",
            ));
        }
        let title = model
            .attribute("name")
            .or_else(|| model.attribute("id"))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "SBML model".to_string());

        // Pass A: find the lists, and refuse what is not translated.
        let mut comp_nodes: Vec<X> = Vec::new();
        let mut species_nodes: Vec<X> = Vec::new();
        let mut param_nodes: Vec<X> = Vec::new();
        let mut rule_nodes: Vec<X> = Vec::new();
        let mut reaction_nodes: Vec<X> = Vec::new();
        for list in model.children().filter(X::is_element) {
            let items = || list.children().filter(X::is_element);
            match local(list) {
                "listOfFunctionDefinitions" => {
                    if let Some(f) = items().next() {
                        return Err(src.err(
                            f.range().start,
                            Kind::Unsupported,
                            "function definitions (`functionDefinition`) are not translated; \
                             write the function body into the kinetic law",
                        ));
                    }
                }
                "listOfInitialAssignments" => {
                    if let Some(f) = items().next() {
                        return Err(src.err(
                            f.range().start,
                            Kind::Unsupported,
                            "initial assignments (`initialAssignment`) are not translated; \
                             give the value in the element instead",
                        ));
                    }
                }
                "listOfEvents" => {
                    if let Some(f) = items().next() {
                        return Err(src.err(
                            f.range().start,
                            Kind::Unsupported,
                            "events (`event`) are not translated",
                        ));
                    }
                }
                "listOfConstraints" => {
                    if items().next().is_some() {
                        self.warnings
                            .push("the constraints in the file are ignored".to_string());
                    }
                }
                "listOfCompartments" => comp_nodes.extend(items()),
                "listOfSpecies" => species_nodes.extend(items()),
                "listOfParameters" => param_nodes.extend(items()),
                "listOfRules" => rule_nodes.extend(items()),
                "listOfReactions" => reaction_nodes.extend(items()),
                _ => {}
            }
        }

        // Pass B: compartments, species, parameters, and every id in the model.
        let mut ids: HashMap<String, &'static str> = HashMap::new();
        let mut claim = |id: &str, what: &'static str, at: usize| -> Result<()> {
            if let Some(prev) = ids.insert(id.to_string(), what) {
                return Err(src.err(
                    at,
                    Kind::NameCollision,
                    format!("the id `{id}` is used by a {prev} and by a {what}"),
                ));
            }
            Ok(())
        };

        let mut compartments: Vec<Compartment> = Vec::new();
        for n in &comp_nodes {
            if local(*n) != "compartment" {
                continue;
            }
            let id = src.id(*n)?;
            claim(&id, "compartment", n.range().start)?;
            if src.num(*n, "spatialDimensions")? == Some(0.0) {
                return Err(src.err(
                    n.range().start,
                    Kind::Unsupported,
                    format!("the compartment `{id}` has no spatial dimensions"),
                ));
            }
            compartments.push(Compartment {
                id,
                size: src.num(*n, "size")?,
            });
        }

        let mut species: Vec<Species> = Vec::new();
        for n in &species_nodes {
            if local(*n) != "species" {
                continue;
            }
            let id = src.id(*n)?;
            claim(&id, "species", n.range().start)?;
            if n.attribute("conversionFactor").is_some() {
                return Err(src.err(
                    n.range().start,
                    Kind::Unsupported,
                    format!("the `conversionFactor` of species `{id}` is not translated"),
                ));
            }
            let amount = src.num(*n, "initialAmount")?;
            let conc = src.num(*n, "initialConcentration")?;
            if amount.is_some() && conc.is_some() {
                return Err(src.err(
                    n.range().start,
                    Kind::Mismatch,
                    format!(
                        "the species `{id}` has both `initialAmount` and `initialConcentration`"
                    ),
                ));
            }
            species.push(Species {
                compartment: src.req(*n, "compartment")?,
                amount,
                conc,
                only_substance: src.flag(*n, "hasOnlySubstanceUnits")?.unwrap_or(false),
                boundary: src.flag(*n, "boundaryCondition")?.unwrap_or(false),
                constant: src.flag(*n, "constant")?,
                at: n.range().start,
                id,
            });
        }

        let mut params: Vec<Param> = Vec::new();
        for n in &param_nodes {
            if local(*n) != "parameter" {
                continue;
            }
            let id = src.id(*n)?;
            claim(&id, "parameter", n.range().start)?;
            params.push(Param {
                value: src.num(*n, "value")?,
                constant: src.flag(*n, "constant")?,
                at: n.range().start,
                id,
            });
        }

        let mut sref_ids: HashSet<String> = HashSet::new();
        for n in &reaction_nodes {
            if local(*n) != "reaction" {
                continue;
            }
            let id = src.id(*n)?;
            claim(&id, "reaction", n.range().start)?;
            for list in n.children().filter(X::is_element) {
                if matches!(local(list), "listOfReactants" | "listOfProducts") {
                    for r in list.children().filter(X::is_element) {
                        if let Some(rid) = r.attribute("id") {
                            claim(rid.trim(), "species reference", r.range().start)?;
                            sref_ids.insert(rid.trim().to_string());
                        }
                    }
                }
            }
        }

        // Pass C: reactions and rules.
        let mut taken: HashSet<String> = ids.keys().map(|k| k.to_ascii_lowercase()).collect();
        let mut reactions: Vec<Reaction> = Vec::new();
        for n in &reaction_nodes {
            if local(*n) != "reaction" {
                continue;
            }
            reactions.push(self.reaction(*n, l3, &mut taken)?);
        }

        let mut rules: Vec<RuleDef> = Vec::new();
        let mut rule_of: HashMap<String, usize> = HashMap::new();
        for n in &rule_nodes {
            let rate = match local(*n) {
                "assignmentRule" => false,
                "rateRule" => true,
                "algebraicRule" => {
                    return Err(src.err(
                        n.range().start,
                        Kind::Unsupported,
                        "algebraic rules (`algebraicRule`) are not translated",
                    ));
                }
                other => {
                    return Err(src.err(
                        n.range().start,
                        Kind::Unsupported,
                        format!("the rule `{other}` is not translated"),
                    ));
                }
            };
            let target = src.req(*n, "variable")?;
            let Some(m) = n.children().find(|c| c.is_element() && local(*c) == "math") else {
                return Err(src.err(
                    n.range().start,
                    Kind::MissingValue,
                    format!("the rule for `{target}` has no `math`"),
                ));
            };
            let math = mathml::convert(m, &|s: &str| s.to_string(), &mut self.warnings)?;
            if rule_of.insert(target.clone(), rules.len()).is_some() {
                return Err(src.err(
                    n.range().start,
                    Kind::Mismatch,
                    format!("two rules change `{target}`"),
                ));
            }
            rules.push(RuleDef {
                target,
                rate,
                math: Some(math),
                at: n.range().start,
            });
        }

        // Checks across the lists.
        let comp_ids: HashSet<&str> = compartments.iter().map(|c| c.id.as_str()).collect();
        let species_ids: HashSet<&str> = species.iter().map(|s| s.id.as_str()).collect();
        let param_ids: HashSet<&str> = params.iter().map(|p| p.id.as_str()).collect();
        for s in &species {
            if !comp_ids.contains(s.compartment.as_str()) {
                return Err(src.err(
                    s.at,
                    Kind::Undeclared,
                    format!(
                        "the species `{}` is in the compartment `{}`, which the file does not define",
                        s.id, s.compartment
                    ),
                ));
            }
        }
        for r in &reactions {
            for (sp, _) in &r.terms {
                if !species_ids.contains(sp.as_str()) {
                    return Err(src.err(
                        r.at,
                        Kind::Undeclared,
                        format!("the reaction `{}` uses the species `{sp}`, which the file does not define", r.id),
                    ));
                }
            }
        }
        for rule in &rules {
            let t = rule.target.as_str();
            if comp_ids.contains(t) {
                return Err(src.err(
                    rule.at,
                    Kind::Unsupported,
                    format!(
                        "a rule changes the size of the compartment `{t}`, which is not translated"
                    ),
                ));
            }
            if sref_ids.contains(t) {
                return Err(src.err(
                    rule.at,
                    Kind::Unsupported,
                    format!("a rule changes the stoichiometry `{t}`, which is not translated"),
                ));
            }
            if !species_ids.contains(t) && !param_ids.contains(t) {
                return Err(src.err(
                    rule.at,
                    Kind::Undeclared,
                    format!("a rule changes `{t}`, which is not a species or a parameter"),
                ));
            }
        }

        // Which species do reactions change, and by how much.
        let mut part: HashMap<&str, Vec<(&str, f64)>> = HashMap::new();
        for r in &reactions {
            for (sp, c) in &r.terms {
                part.entry(sp.as_str())
                    .or_default()
                    .push((r.id.as_str(), *c));
            }
        }

        let size_of: HashMap<&str, Option<f64>> = compartments
            .iter()
            .map(|c| (c.id.as_str(), c.size))
            .collect();

        let mut fixed: Vec<(String, Expr)> = Vec::new();
        let mut states: Vec<State> = Vec::new();
        let mut cands: Vec<Candidate> = Vec::new();
        let mut rate_params: Vec<(usize, Expr)> = Vec::new();

        for c in &compartments {
            cands.push(Candidate {
                name: c.id.clone(),
                value: c.size,
                what: "compartment size",
                at: None,
            });
        }
        // Parameters: assignment targets become fixed quantities, rate targets states.
        for (i, p) in params.iter().enumerate() {
            match rule_of.get(&p.id).copied() {
                None => cands.push(Candidate {
                    name: p.id.clone(),
                    value: p.value,
                    what: "parameter",
                    at: Some(p.at),
                }),
                Some(ri) => {
                    if p.constant == Some(true) {
                        return Err(src.err(
                            p.at,
                            Kind::Mismatch,
                            format!("the parameter `{}` is constant and has a rule", p.id),
                        ));
                    }
                    if rules[ri].rate {
                        let math = rules[ri].math.take().expect("each rule is used once");
                        rate_params.push((i, math));
                    }
                }
            }
        }

        // Species.
        for s in &species {
            let participates = part.contains_key(s.id.as_str());
            let in_reaction_as_variable = participates && !s.boundary;
            let rule = rule_of.get(&s.id).copied();
            if s.constant == Some(true) {
                if rule.is_some() {
                    return Err(src.err(
                        s.at,
                        Kind::Mismatch,
                        format!("the species `{}` is constant and has a rule", s.id),
                    ));
                }
                if in_reaction_as_variable {
                    return Err(src.err(
                        s.at,
                        Kind::Mismatch,
                        format!(
                            "the species `{}` is constant, not a boundary species, and takes part in a reaction",
                            s.id
                        ),
                    ));
                }
            }
            let size = size_of.get(s.compartment.as_str()).copied().flatten();
            match rule {
                Some(ri) => {
                    if in_reaction_as_variable {
                        return Err(src.err(
                            s.at,
                            Kind::Mismatch,
                            format!(
                                "the species `{}` has a rule and also takes part in a reaction; \
                                 SBML requires `boundaryCondition=\"true\"` for that",
                                s.id
                            ),
                        ));
                    }
                    let math = rules[ri].math.take().expect("each rule is used once");
                    if rules[ri].rate {
                        let init = self.species_value(s, size)?.ok_or_else(|| {
                            src.err(
                                s.at,
                                Kind::MissingValue,
                                format!(
                                    "the species `{}` has a rate rule and no initial value",
                                    s.id
                                ),
                            )
                        })?;
                        states.push(State {
                            name: s.id.clone(),
                            rhs: math,
                            init: Some(init),
                        });
                    } else {
                        fixed.push((s.id.clone(), math));
                    }
                }
                None if in_reaction_as_variable => {
                    let init = self.species_value(s, size)?.ok_or_else(|| {
                        src.err(
                            s.at,
                            Kind::MissingValue,
                            format!(
                                "the species `{}` has no initial amount or concentration",
                                s.id
                            ),
                        )
                    })?;
                    let mut terms: Vec<Expr> = Vec::new();
                    let mut net: Vec<(&str, f64)> = Vec::new();
                    for (rid, c) in &part[s.id.as_str()] {
                        match net.iter_mut().find(|(r, _)| r == rid) {
                            Some(slot) => slot.1 += c,
                            None => net.push((rid, *c)),
                        }
                    }
                    for (rid, c) in net {
                        if c == 0.0 {
                            continue;
                        }
                        let v = Expr::var(rid);
                        terms.push(if c == 1.0 {
                            v
                        } else if c == -1.0 {
                            Expr::negate(v)?
                        } else {
                            Expr::bin(Op::Mul, Expr::num(c)?, v)?
                        });
                    }
                    let mut rhs = match mathml::balanced(Op::Add, terms)? {
                        Some(e) => e,
                        None => Expr::num(0.0)?,
                    };
                    if !s.only_substance {
                        rhs = Expr::bin(Op::Div, rhs, Expr::var(s.compartment.as_str()))?;
                    }
                    states.push(State {
                        name: s.id.clone(),
                        rhs,
                        init: Some(init),
                    });
                }
                None => {
                    if !s.boundary && s.constant != Some(true) && !participates {
                        self.warnings.push(format!(
                            "the species `{}` is in no reaction and has no rule, so it stays constant",
                            s.id
                        ));
                    }
                    cands.push(Candidate {
                        name: s.id.clone(),
                        value: self.species_value(s, size)?,
                        what: "species",
                        at: Some(s.at),
                    });
                }
            }
        }

        for (i, math) in rate_params {
            let p = &params[i];
            let Some(v) = p.value else {
                return Err(src.err(
                    p.at,
                    Kind::MissingValue,
                    format!("the parameter `{}` has a rate rule and no value", p.id),
                ));
            };
            states.push(State {
                name: p.id.clone(),
                rhs: math,
                init: Some(v),
            });
        }
        // Assignment rules on parameters.
        for p in &params {
            if let Some(&ri) = rule_of.get(&p.id)
                && !rules[ri].rate
            {
                let math = rules[ri].math.take().expect("each rule is used once");
                fixed.push((p.id.clone(), math));
            }
        }

        // Reactions become fixed quantities. Local parameters join the candidates.
        let mut locals: Vec<Local> = Vec::new();
        let mut reaction_fixed: Vec<(String, Expr)> = Vec::new();
        for r in reactions {
            locals.extend(r.locals);
            reaction_fixed.push((r.id, r.math));
        }
        reaction_fixed.append(&mut fixed);
        let fixed = reaction_fixed;

        // Which names do the expressions read?
        let mut referenced: HashSet<String> = HashSet::new();
        for (_, e) in &fixed {
            e.visit_vars(&mut |v| {
                referenced.insert(v.to_string());
            });
        }
        for s in &states {
            s.rhs.visit_vars(&mut |v| {
                referenced.insert(v.to_string());
            });
        }
        for l in locals {
            if referenced.contains(&l.name) {
                cands.push(Candidate {
                    name: l.name,
                    value: l.value,
                    what: "local parameter",
                    at: Some(l.at),
                });
            }
        }

        let mut par: Vec<(String, f64)> = Vec::new();
        for c in cands {
            match c.value {
                Some(v) => par.push((c.name, v)),
                None if referenced.contains(&c.name) => {
                    let msg = format!(
                        "the {} `{}` has no value, and the model uses it",
                        c.what, c.name
                    );
                    return Err(match c.at {
                        Some(at) => src.err(at, Kind::MissingValue, msg),
                        None => ImportError::new(Kind::MissingValue, msg),
                    });
                }
                None => self.warnings.push(format!(
                    "the {} `{}` has no value and nothing uses it, so it is left out",
                    c.what, c.name
                )),
            }
        }

        let fixed = sort_fixed(fixed)?;
        let model = OdeModel {
            title,
            params: par,
            fixed,
            states,
        };
        let mut out = emit(&model)?;
        let mut all = self.warnings;
        all.append(&mut out.warnings);
        out.warnings = all;
        Ok(out)
    }

    /// The initial value of a species as the quantity the state holds (a concentration unless
    /// `hasOnlySubstanceUnits`). `None` when the file gives no value.
    fn species_value(&self, s: &Species, size: Option<f64>) -> Result<Option<f64>> {
        let need = || {
            size.ok_or_else(|| {
                self.src.err(
                    s.at,
                    Kind::MissingValue,
                    format!(
                        "the species `{}` needs the size of compartment `{}` to convert its \
                         initial value, and the file gives none",
                        s.id, s.compartment
                    ),
                )
            })
        };
        let v = match (s.only_substance, s.amount, s.conc) {
            (_, None, None) => return Ok(None),
            (false, Some(a), _) => a / need()?,
            (false, None, Some(c)) | (true, Some(c), _) => c,
            (true, None, Some(c)) => c * need()?,
        };
        if !v.is_finite() {
            return Err(self.src.err(
                s.at,
                Kind::Mismatch,
                format!(
                    "the initial value of `{}` is not finite (compartment size 0?)",
                    s.id
                ),
            ));
        }
        Ok(Some(v))
    }

    fn reaction(&mut self, n: X, l3: bool, taken: &mut HashSet<String>) -> Result<Reaction> {
        let src = self.src;
        let id = src.id(n)?;
        let at = n.range().start;
        if src.flag(n, "fast")? == Some(true) {
            return Err(src.err(
                at,
                Kind::Unsupported,
                format!("the reaction `{id}` is marked `fast`, which is not translated"),
            ));
        }
        let mut terms: Vec<(String, f64)> = Vec::new();
        for list in n.children().filter(X::is_element) {
            let sign = match local(list) {
                "listOfReactants" => -1.0,
                "listOfProducts" => 1.0,
                _ => continue,
            };
            for r in list.children().filter(X::is_element) {
                if local(r) != "speciesReference" {
                    continue;
                }
                if r.children()
                    .any(|c| c.is_element() && local(c) == "stoichiometryMath")
                {
                    return Err(src.err(
                        r.range().start,
                        Kind::Unsupported,
                        "`stoichiometryMath` is not translated",
                    ));
                }
                if r.attribute("denominator").is_some() {
                    return Err(src.err(
                        r.range().start,
                        Kind::Unsupported,
                        "a `denominator` on a species reference is not translated",
                    ));
                }
                let species = src.req(r, "species")?;
                let stoich = match src.num(r, "stoichiometry")? {
                    Some(v) => v,
                    None if l3 => {
                        return Err(src.err(
                            r.range().start,
                            Kind::MissingValue,
                            format!(
                                "the species reference to `{species}` in reaction `{id}` has no stoichiometry"
                            ),
                        ));
                    }
                    None => 1.0,
                };
                terms.push((species, sign * stoich));
            }
        }

        let Some(law) = n
            .children()
            .find(|c| c.is_element() && local(*c) == "kineticLaw")
        else {
            return Err(src.err(
                at,
                Kind::MissingValue,
                format!("the reaction `{id}` has no kinetic law"),
            ));
        };
        let mut local_map: HashMap<String, String> = HashMap::new();
        let mut locals: Vec<Local> = Vec::new();
        for list in law.children().filter(X::is_element) {
            if !matches!(local(list), "listOfParameters" | "listOfLocalParameters") {
                continue;
            }
            for p in list.children().filter(X::is_element) {
                if !matches!(local(p), "parameter" | "localParameter") {
                    continue;
                }
                let lid = src.id(p)?;
                if local_map.contains_key(&lid) {
                    return Err(src.err(
                        p.range().start,
                        Kind::NameCollision,
                        format!("the reaction `{id}` has two local parameters called `{lid}`"),
                    ));
                }
                let mut name = format!("{id}_{lid}");
                while taken.contains(&name.to_ascii_lowercase()) {
                    name.push('_');
                }
                taken.insert(name.to_ascii_lowercase());
                local_map.insert(lid, name.clone());
                locals.push(Local {
                    name,
                    value: src.num(p, "value")?,
                    at: p.range().start,
                });
            }
        }
        let Some(m) = law
            .children()
            .find(|c| c.is_element() && local(*c) == "math")
        else {
            return Err(src.err(
                law.range().start,
                Kind::MissingValue,
                format!("the kinetic law of reaction `{id}` has no `math`"),
            ));
        };
        let rename = |s: &str| local_map.get(s).cloned().unwrap_or_else(|| s.to_string());
        let math = mathml::convert(m, &rename, &mut self.warnings)?;
        Ok(Reaction {
            id,
            terms,
            math,
            locals,
            at,
        })
    }
}

/// Orders fixed quantities so each follows the ones it reads. Ties keep the input order.
fn sort_fixed(fixed: Vec<(String, Expr)>) -> Result<Vec<(String, Expr)>> {
    let n = fixed.len();
    let index: HashMap<&str, usize> = fixed
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.as_str(), i))
        .collect();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    for (i, (_, e)) in fixed.iter().enumerate() {
        let mut deps: BTreeSet<usize> = BTreeSet::new();
        e.visit_vars(&mut |v| {
            if let Some(&j) = index.get(v) {
                deps.insert(j);
            }
        });
        for j in deps {
            dependents[j].push(i);
            indeg[i] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &d in &dependents[i] {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                ready.insert(d);
            }
        }
    }
    if order.len() < n {
        let stuck: Vec<&str> = (0..n)
            .filter(|&i| indeg[i] > 0)
            .take(5)
            .map(|i| fixed[i].0.as_str())
            .collect();
        return Err(ImportError::new(
            Kind::Mismatch,
            format!(
                "these quantities depend on each other in a cycle: {}",
                stuck.join(", ")
            ),
        ));
    }
    let mut slots: Vec<Option<(String, Expr)>> = fixed.into_iter().map(Some).collect();
    Ok(order
        .into_iter()
        .map(|i| slots[i].take().expect("each index appears once"))
        .collect())
}
