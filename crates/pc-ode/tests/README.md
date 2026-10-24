# Tests for `pc-ode`

This folder holds the integration tests, the model corpus and the invalid files. The unit tests sit next to the code, in `src/**/tests.rs`.

## Layout

| Path | What it holds |
|---|---|
| `corpus/*.ode` | Models written from textbook or paper equations. Each has a golden file. |
| `corpus/include/` | Files that only another corpus file includes. The `*.ode` glob does not reach them. |
| `corpus/*.golden` | The expected diagnostics and model summary of the matching `.ode` file. |
| `invalid/odeNNNN.ode` | One file per diagnostic code. The first line is a comment `# ODEnnnn ...` naming the code. A second file for a code is `odeNNNN_b.ode`. |
| `invalid/include/` | Helper files for the include codes (cycle, depth, count, size). |
| `common/mod.rs` | Round trip helpers used by `round_trip.rs` and `corpus.rs`. |
| `corpus.rs` | Corpus tests: load, round trip, golden files, the two chain files. |
| `invalid.rs` | Checks that every code in `CODES` has a file that produces it. |
| `round_trip.rs` | Writer unit tests and the `proptest` round trip properties. |

Every corpus model starts with a comment naming the model and its source. Parameter values that are not from that source are marked as illustrative. No corpus file is copied from the XPPAUT examples.

## Golden files

`foo.golden` is plain text with LF line endings and no absolute paths or dates.

```
diagnostics
<file>:<line>:<col>: <severity>[ODEnnnn]: <message>
  note: <file>:<line>:<col>: <text>
  fix-it: <file>:<line>:<col>: replace "<old>" with "<new>"
summary
kind: Ode
smooth: true
simulation_only: false
states: 2
parameters: 3
numbers: 0
derived: 0
fixed: 0
functions: 0
aux: 0
tables: 0
wiener: 0
symbols
<kind> <name>
```

`<file>` is the path relative to `tests/corpus/` with `/` separators, so a diagnostic in an included file names `include/...`. Lines and columns are 1-based and columns count characters. Diagnostics keep the order `load` returns. Symbols are in declaration order; `<kind>` is `state-ode`, `state-map`, `state-markov`, `parameter`, `number`, `derived`, `fixed`, `aux`, `function(<params>)`, `table` or `wiener`.

A mismatch fails the test and names the first differing line. A missing golden file fails too. To rewrite the golden files after a change that is meant to alter them:

```
PC_BLESS=1 cargo test -p pc-ode --test corpus
```

Read the diff before you commit it.

## Adding a case

- A corpus model: add `corpus/name.ode` with a source comment, run the bless command, and read the new `.golden`. If the model needs warnings, add its codes to `EXPECTED_WARNINGS` in `corpus.rs`.
- An invalid file: add `invalid/odeNNNN_b.ode` starting with `# ODENNNN`. The test finds it without a code change.

One code, ODE0018 (old-style file), has no file. The check fires only when the first line of the file starts with a digit, and every invalid file must start with its naming comment. The input is built in `invalid.rs` (`BUILT_IN`) instead.

## Coverage matrix

Each row of the table and each rule in section 2 of `docs/SPEC.md` is mapped to tests. A name of the form `path::test` is a test function in that file. A name in `corpus/` or `invalid/` is a file in this folder. Paths that start with `src/` are unit tests and the others are in `crates/pc-ode/tests/`.

### Parsing rules

| Rule | Tests and files |
|---|---|
| Keywords: first letter only for the nine declarations | `src/parser/stmt/tests.rs::every_keyword_abbreviation`, `src/parser/stmt/tests.rs::keyword_letter_names_are_formulas`, `round_trip.rs::round_trip_keyword_letter_names` |
| Keywords: full word for `number`, `set`, `options`, `only`, `special`, `solv`, `export`, `volt`, `hom_bcs`; `p=3` is a fixed quantity | `src/parser/stmt/tests.rs::keyword_letter_names_are_formulas`, `src/parser/stmt/tests.rs::unsupported_families`, `src/parser/stmt/tests.rs::options_and_set`, `corpus/unsupported.ode`, `corpus/sets_include.ode` |
| Case is ignored | `src/check/tests.rs::case_is_ignored`, `src/lexer.rs::names_keep_dots_and_fold_case`, `corpus/abbreviations.ode` |
| Names over 10 characters warn | `src/check/tests.rs::long_names`, `src/check/tests.rs::long_name_notes`, `invalid/ode0013_b.ode` |
| Line continuation | `src/lexer.rs::continuation_is_a_blank`, `src/parser/stmt/tests.rs::continued_option_is_not_spaced`, `corpus/abbreviations.ode` |
| Order: names usable before declaration, fixed quantities in file order | `src/check/tests.rs::fixed_quantity_forward_reference`, `src/check/tests.rs::fixed_forward_through_functions`, `src/check/tests.rs::undefined_names`, `invalid/ode0042.ode`, `invalid/ode0039.ode` |
| Spaces around `=` in lists and formulas; ignored option warns | `src/parser/stmt/tests.rs::entry_lists`, `src/parser/stmt/tests.rs::spaced_option`, `invalid/ode0025.ode` |
| Declared values are signed number literals; a repeated `par` is an error | `src/parser/stmt/tests.rs::bad_values`, `src/parser/stmt/tests.rs::literal_values_keep_their_text`, `src/parser/stmt/tests.rs::spaced_sign_is_one_bad_value`, `src/check/tests.rs::duplicates`, `invalid/ode0023.ode`, `invalid/ode0041.ode` |
| Comments: `#` at line start, trailing comment after `par`, `number`, `@`, error elsewhere | `src/parser/stmt/tests.rs::trailing_comments_after_par_number_and_options_are_kept`, `src/parser/stmt/tests.rs::trailing_comment_is_an_error_but_the_statement_stays`, `round_trip.rs::trailing_comments_are_kept`, `invalid/ode0022.ode` |
| Arrays: range lines and `%[1..n]` blocks | `src/expand/tests.rs::chain_as_range_lines_and_as_a_block`, `src/expand/tests.rs::blocks_copy_their_body_per_index`, `corpus.rs::chain_range_and_block_declare_the_same_model`, `corpus/chain_range.ode`, `corpus/chain_block.ode`, `corpus/cable.ode` |
| Old-style files (first line starts with a digit) | `src/parser/stmt/tests.rs::old_style_file`, `invalid.rs::BUILT_IN` (ODE0018) |

### Constructs

| Construct | Tests and files |
|---|---|
| ODE (`x'=`, `dx/dt=`) | `src/parser/stmt/tests.rs::formulas`, `src/parser/stmt/tests.rs::ddt_needs_a_letter_after_d_and_no_index`, `corpus/lorenz.ode`, `corpus/abbreviations.ode` |
| Map (`x(t+1)=`) | `src/check/tests.rs::model_kind`, `round_trip.rs::map_statements_round_trip`, `corpus.rs::model_kinds_match_the_equations`, `corpus/henon_map.ode`, `corpus/logistic_map.ode`, `invalid/ode0052.ode` |
| Initial data (`init`, `x(0)=`) | `src/check/tests.rs::initial_values`, `corpus/fitzhugh_nagumo.ode`, `corpus/chain_range.ode`, `invalid/ode0044.ode`, `invalid/ode0045.ode` |
| Parameters (`par`) | `src/parser/stmt/tests.rs::entry_lists`, `corpus/lorenz.ode`, `invalid/ode0041.ode` |
| Numbers and derived parameters (`number`, `!d=`) | `src/check/tests.rs::derived_parameters`, `corpus/abbreviations.ode`, `invalid/ode0053.ode` |
| Fixed quantities | `src/check/tests.rs::fixed_quantity_forward_reference`, `src/check/tests.rs::circular_definitions`, `corpus/morris_lecar.ode`, `corpus/hodgkin_huxley.ode`, `invalid/ode0042.ode`, `invalid/ode0043.ode` |
| Auxiliary output (`aux`) | `src/check/tests.rs::aux_in_formulas`, `corpus/hodgkin_huxley.ode`, `invalid/ode0051.ode` |
| User functions | `src/check/tests.rs::function_parameters`, `src/check/tests.rs::recursive_functions`, `src/check/tests.rs::argument_counts`, `src/check/tests.rs::arg_names`, `corpus/hodgkin_huxley.ode`, `invalid/ode0046.ode`, `invalid/ode0047.ode`, `invalid/ode0048.ode`, `invalid/ode0049.ode`, `invalid/ode0050.ode` |
| Tables: inline and file | `src/parser/stmt/tests.rs::table_forms`, `src/parser/stmt/tests.rs::table_fields_are_named`, `corpus/inline_table.ode`, `corpus/unsupported.ode`, `invalid/ode0036.ode` |
| Options (`@`) | `src/parser/stmt/tests.rs::options_and_set`, `src/parser/stmt/tests.rs::unknown_option`, `src/options.rs::documented_names_are_known`, `corpus/options_heavy.ode`, `invalid/ode0024.ode` |
| Include and `set` | `src/include/tests.rs::one_include`, `src/include/tests.rs::nested_includes`, `src/include/tests.rs::fs_loader_resolves_against_the_including_file`, `src/parser/stmt/tests.rs::set_and_global_brace_lists`, `corpus/sets_include.ode`, `invalid/ode0059.ode` to `invalid/ode0064.ode` |
| Arrays (expanded) | `src/expand/tests.rs::every_index_form`, `src/expand/tests.rs::ranges_count_down`, `src/expand/tests.rs::the_copy_cap`, `corpus/chain_range.ode`, `corpus/cable.ode`, `invalid/ode0065.ode` to `invalid/ode0071.ode` |
| Global flags | `src/parser/stmt/tests.rs::global_markov_wiener_include_comment`, `src/parser/stmt/tests.rs::global_condition_may_be_in_braces`, `src/check/tests.rs::global_targets`, `corpus/stochastic.ode`, `invalid/ode0058.ode` |
| Markov and Wiener | `src/parser/stmt/tests.rs::markov_rows`, `src/parser/stmt/tests.rs::markov_lines_keep_source_order`, `src/check/tests.rs::markov_states_do_not_set_the_kind`, `corpus/stochastic.ode`, `invalid/ode0026.ode`, `invalid/ode0070.ode` |
| `delay` | `src/check/tests.rs::delay_shift_and_del_shft`, `src/check/tests.rs::delay_with_one_argument`, `corpus/unsupported.ode`, `invalid/ode0054.ode` |
| Volterra and boundary value problems | `src/parser/stmt/tests.rs::unsupported_families`, `corpus/unsupported.ode`, `invalid/ode0028.ode`, `invalid/ode0029.ode` |
| DAEs (`0=`, `solv`) | `src/parser/stmt/tests.rs::unsupported_families`, `corpus/unsupported.ode`, `invalid/ode0030.ode` |
| Other constructs (`special`, `export`, `only`, `options`, `shift`, `del_shft`, `sum(..)of(..)`, `hom_bcs`) | `src/parser/stmt/tests.rs::unsupported_families`, `src/parser/stmt/tests.rs::delay_shift_and_del_shft_stay_calls`, `src/check/tests.rs::unsupported_statements_that_may_declare_names`, `corpus/unsupported.ode`, `invalid/ode0031.ode` to `invalid/ode0038.ode`, `invalid/ode0055.ode` to `invalid/ode0057.ode` |
| End (`done`) | `src/parser/stmt/tests.rs::done_ends_the_file`, `src/include/tests.rs::done_ends_the_included_file_only`, `corpus.rs::every_corpus_file_has_a_source_comment_and_ends_with_done` |

### Expressions

| Rule | Tests and files |
|---|---|
| Operators and XPPAUT precedence, with warnings where the reading is surprising | `src/parser/expr/tests.rs::measured_precedence_table`, `src/parser/expr/tests.rs::levels_are_left_associative`, `src/parser/expr/tests.rs::precedence_warning_spans_and_fix_its`, `invalid/ode0014.ode`, `invalid/ode0010.ode`, `invalid/ode0011.ode` |
| Implicit multiplication (`2v`) with a portability warning | `src/parser/expr/tests.rs::implicit_multiplication_binds_at_level_two`, `src/parser/expr/tests.rs::implicit_multiplication_warning`, `round_trip.rs::implicit_multiplication_is_written_as_juxtaposition`, `corpus/abbreviations.ode`, `invalid/ode0013.ode` |
| Builtin functions and their arity; `ceil` warns | `src/builtins.rs::lookup_and_special_entries`, `src/check/tests.rs::argument_counts`, `src/check/tests.rs::ceil_portability`, `corpus/morris_lecar.ode`, `corpus/hodgkin_huxley.ode`, `invalid/ode0048.ode` |
| `ran`, `normal`, `poisson`, `markov`, `wiener`, `global` make a model simulation only | `src/check/tests.rs::simulation_only_flag`, `corpus.rs::simulation_only_and_smooth_flags_follow_the_constructs`, `corpus/stochastic.ode` |
| Builtin semantics table agrees with `docs/language.md` | `src/builtins.rs::table_matches_language_md` |
| `if(c)then(a)else(b)` | `src/parser/expr/tests.rs::if_is_an_atom`, `src/parser/expr/tests.rs::malformed_if_is_one_error`, `invalid/ode0016.ode` |

### Non-smooth and simulation-only flags

| Rule | Tests and files |
|---|---|
| `heav sign abs mod flr ceil min max if`, comparisons, `&`, `|`, `not` and tables mark a model non-smooth | `src/check/tests.rs::smooth_flag`, `src/check/tests.rs::comparisons_and_logic_are_not_smooth`, `src/check/tests.rs::if_in_an_uncalled_function_is_not_smooth`, `corpus.rs::simulation_only_and_smooth_flags_follow_the_constructs`, `corpus/inline_table.ode` |
| Smooth models stay flagged smooth | `corpus.rs::simulation_only_and_smooth_flags_follow_the_constructs`, `corpus/lorenz.ode`, `corpus/hodgkin_huxley.ode` |

### Diagnostics and round trip

| Rule | Tests and files |
|---|---|
| Each diagnostic has a code, a line and column, a message and an optional fix-it | `src/diag.rs::code_table_is_unique_and_sorted`, `invalid.rs::every_code_has_an_invalid_file_that_produces_it`, `corpus.rs::corpus_files_match_their_golden_files`, `src/parser/expr/tests.rs::applying_a_fix_it_removes_the_diagnostic` |
| `parse(write(parse(s)))` equals `parse(s)` | `round_trip.rs::round_trip_generated_models`, `round_trip.rs::round_trip_survives_surface_noise`, `corpus.rs::corpus_files_round_trip_through_the_writer`, `corpus.rs::written_corpus_text_loads_with_the_same_diagnostics` |

## Limits

`limits.rs` tests the caps on work and output. Source of each cap and the test that covers it:

| Limit | Value | Test |
|---|---|---|
| `#include` statements reached in one load (ODE0064) | 4096 | `src/include/tests.rs`, `invalid/ode0064.ode` |
| Statements added by included files (ODE0072) | 100000 | `limits.rs::small_file_included_many_times_stops_at_the_statement_cap`, `invalid/ode0072.ode` |
| Text added by included files (ODE0072) | 1 MiB | `limits.rs::text_cap_stops_a_few_long_statements`, `invalid/include/bulk/` |
| Diagnostics per pass and per load (ODE0073) | 10000 | `limits.rs::load_keeps_at_most_the_limit_and_says_how_many_more`, `check_alone_keeps_at_most_the_limit`, `expand_alone_keeps_at_most_the_limit`, `include_pass_alone_keeps_at_most_the_limit`, `invalid/ode0073.ode` |
| Copies of one array line collapse while diagnostics arrive | none | `limits.rs::copies_of_one_line_do_not_fill_the_list`, `a_full_list_still_collapses_nothing_wrongly` |
