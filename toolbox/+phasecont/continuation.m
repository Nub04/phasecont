function br = continuation(m, start, options)
%CONTINUATION Follow a branch of equilibria or cycles, or a curve of special points.
%   BR = phasecont.continuation(M, START, Free = "r") continues START along
%   the free parameter. START is a phasecont.Point or a state vector. The
%   call blocks while the engine runs, polls it (first after 1 ms, then at
%   growing intervals up to 50 ms) and returns a phasecont.Branch. Ctrl+C
%   cancels the run and returns nothing.
%
%   BRS = phasecont.continuation(M, START, Free = "r", Switch = "BP1")
%   follows the branch and then the branch that starts at its first branch
%   point. BRS is a 1-by-2 array: the followed branch, then the switched one.
%
%   C = phasecont.continuation(M, START, Type = "LP", Free = ["a" "b"])
%   follows the curve of fold points through START, which must be an LP
%   point (or a state vector and Params that are one). Type = "H" does the
%   same for Hopf points. Free names two different parameters [a b]; C.Alpha
%   holds [a b] at each point and the codim-2 points (CP, BT, ZH for "LP";
%   GH, BT, ZH, HH for "H") are in C.Specials. Detect, Switch and Var apply
%   to equilibrium runs only.
%
%   LC = phasecont.continuation(M, START, Type = "LC", Free = "r", NTST = 40,
%   NCOL = 4) follows the limit cycles that start at START, either a Hopf
%   point of an equilibrium branch (or a state vector and Params that are
%   one) or a PD point of an earlier cycle branch, where the run follows the
%   cycles of twice the period. NTST is the number of mesh intervals and NCOL
%   the number of collocation points in each. LC.Period holds the period,
%   LC.Eigenvalues the Floquet multipliers and LC.Specials the LPC, PD and NS
%   points; LC.cycle(i) returns the orbit at point i. Detect and Switch do
%   not apply.
%
%   C = phasecont.continuation(M, PT, Type = "PD", Free = ["a" "b"])
%   follows the curve of period-doubling points of cycles through PT, a PD
%   point of an LC branch, with a the free parameter of that branch. Type =
%   "LPC" does the same for an LPC point and Type = "NS" for an NS point.
%   C.Alpha holds [a b], C.Period the period and C.cycle(i) the orbit at
%   point i; an NS curve also has C.Kappa, the cosine of the angle of its
%   critical multipliers. The codim-2 points (LPPD, LPNS, CPC, R1 for
%   "LPC"; LPPD, PDNS, GPD, R2 for "PD"; R1 to R4, LPNS, PDNS, CH for "NS")
%   are in C.Specials. NTST and NCOL default to the mesh of PT, and other
%   values regrid its orbit.
%   Detect, Switch and Var do not apply.
%
%   C = phasecont.continuation(M, LC, Type = "HOM", Index = I, Free = ["a"
%   "b"]) follows the curve of orbits homoclinic to a saddle that starts
%   from point I of LC, an LC branch of long period. The engine moves the
%   slowest point of that orbit to both ends and takes the period as the
%   fixed truncation time T. NTST and NCOL default to those of LC, and other
%   values regrid the orbit. START can instead be a struct with fields t
%   (times, increasing), x (the states at those times, one row per time)
%   and p (the parameters, a struct or a vector) that holds an orbit close
%   to a homoclinic one; T is then t(end) - t(1), the orbit is resampled
%   with pchip on a uniform mesh, and NTST and NCOL are required. C.Alpha
%   holds [a b], C.X0 the saddle, C.Eigenvalues the eigenvalues of the
%   Jacobian at the saddle and C.cycle(i) the orbit at point i. The codim-2
%   points (NSA, BLS, BLU, NSF, NDS, NDU, TLS, TLU, ZES, ZEU, OFS, OFU) are
%   in C.Specials. With InclinationFlips = true the engine also solves an
%   adjoint problem at every point and reports the inclination flips IFS and
%   IFU. Their columns of C.Codim2 are NaN otherwise.
%   Detect, Switch and Var do not apply.
%
%   BR = phasecont.continuation(M, X0, Free = "r", Roots = struct(N = 30,
%   ReMin = -0.5)) follows the equilibria of a delay model, one whose
%   equations call delay(x, tau). Type is "EP", "H" or "LC" for it. N is the
%   degree of the pseudospectral approximation of the characteristic roots and
%   ReMin the smallest real part to look for; ReMin must be below 0, and
%   neither has a default. BR.RootCount holds the number of roots with
%   positive real part at each point and BR.Stable is true where it is 0. A
%   Hopf point (H) is reported where the count changes by a complex pair, a
%   fold (LP) or a branch point (BP) where it changes by one; the Data of the
%   point holds omega and the counts before and after in Data.values. The
%   branch has no Eigenvalues, Tests or Codim2. START must be a state vector
%   or an EP point. Detect takes "LP" and "BP" only (default both). Switch,
%   MeshAdapt and every Type other than "EP", "H" and "LC" are refused. The
%   engine returns the points when the run ends, so a live Plot draws once and
%   a cancelled run keeps nothing.
%
%   C = phasecont.continuation(M, START, Type = "H", Free = ["a" "b"], Omega =
%   w) follows the curve of Hopf points of a delay model. START is an H point
%   of a delay equilibrium branch, which supplies the frequency, or a state
%   vector with Params and the Omega of that Hopf point; Omega with a Point
%   start is an error. C.Alpha holds [a b] at each point, C.U the
%   equilibrium and C.Kappa the square of the frequency. C.Codim2 holds the
%   first Lyapunov coefficient l1 and the Hopf-Hopf test hh, named by
%   C.Codim2Labels = ["GH" "HH"]; C.Eigenvalues, C.Tests and C.RootCount are
%   empty. C.Specials holds BT and ZH always, GH with Lyapunov = true and HH
%   with Roots = struct(N = n, ReMin = r), the root search at every point.
%   Lyapunov is a logical scalar, default false. With Coefficients = true
%   the zero-Hopf coefficients of each ZH special and the Hopf-Hopf
%   coefficients of each HH special arrive in Special.Data.values (HH needs
%   Roots too); a special where the engine refuses has none, and the branch
%   gets a warning. Detect, Switch, MeshAdapt and Var are refused. As for
%   equilibria, the curve arrives when the run ends.
%
%   LC = phasecont.continuation(M, START, Type = "LC", Free = "tau", NTST = 20,
%   NCOL = 4, Omega = w, Eps = 0.05, Floquet = struct(NHist = 12, Count = 4))
%   follows the periodic orbits of a delay model that branch off an H point.
%   START is an H point of a delay equilibrium branch, which supplies the
%   frequency, or a state vector with Params and the Omega of that Hopf point;
%   Omega with a Point start is an error. The first orbit is the Hopf orbit of
%   amplitude Eps, corrected by the engine, and Free is one parameter. NTST,
%   NCOL and Eps have no default. With Floquet the engine also computes the
%   leading Count Floquet multipliers of every orbit on a history grid of NHist
%   intervals, and LC.Eigenvalues holds them; without it that array has no
%   columns. LC.Alpha, LC.U (one orbit per row), LC.Period and LC.cycle(i)
%   work as for an ODE cycle, the mesh is uniform, and LC.Specials holds the
%   LPC, PD and NS points the engine finds, with the orbit in Special.Data.
%   Roots, Lyapunov, Coefficients, Detect, MeshAdapt, Switch and Var are
%   refused, and the branch arrives when the run ends.
%
%   Name-value options
%     Type       "EP" (default), "LP", "H", "LC", "LPC", "PD", "NS" or "HOM"
%     Free       free parameter, one of M.Parameters (required); two names
%                for Type "LP", "H", "LPC", "PD", "NS" or "HOM"
%     NTST NCOL  mesh intervals and collocation points per interval,
%                positive integers (required for Type "LC", also of a delay
%                model, and for "HOM"
%                from a struct; for "LPC", "PD", "NS" and "HOM" from an LC
%                branch they default to those of the start)
%     Index      the point of the LC branch a HOM curve starts from
%                (required for Type "HOM" from a Branch)
%     MaxSteps   most steps per branch
%     Direction  1 or -1, the first direction along the parameter
%     Bounds     [lo hi], finite, the run stops outside them; on a curve
%                they bound the second free parameter
%     H0 Hmin Hmax   first, smallest and largest step length
%     TolX TolF  Newton tolerances on the step and the residual
%     MaxNewton  most Newton iterations per step
%     Detect     labels to detect, a subset of ["LP" "BP" "H"] (default all);
%                Type "EP" only; ["LP" "BP"] only, and the default, for a
%                delay model
%     Switch     special point to switch at, like "BP1"; Type "EP" only
%     Plot       true draws the branch while the run goes and ends with the
%                styled plot of Branch.plot (default false)
%     Var        variable to plot, a name or an index into M.Variables
%                (default the first variable of M); Type "EP" or "LC" (not a
%                delay "LC"), a
%                curve of any other Type is plotted in the plane of its two
%                free parameters
%     Params     struct or vector that overrides the parameters of START
%     InclinationFlips  true adds the inclination flips IFS and IFU to the
%                codim-2 tests of a HOM curve, at the cost of one linear solve
%                per point (logical scalar, default false); Type "HOM" only
%     MeshAdapt  number of continuation steps between two mesh adaptations of
%                a cycle or homoclinic orbit; 0 keeps the start mesh (integer
%                of at least 0, engine default 3); Type "LC", "LPC", "PD",
%                "NS" or "HOM" only
%     Roots      struct(N = n, ReMin = r) for a delay model (required for
%                Type "EP", optional for "H", refused for "LC" and for an ODE
%                model); see above
%     Omega      frequency of the Hopf point a delay "H" curve or "LC" run
%                starts from, a finite real scalar above 1e-8 (required for a
%                state vector start, an error with a Point); Type "H" or "LC"
%                of a delay model only
%     Eps        amplitude of the first orbit of a delay "LC" run, a finite
%                real scalar above 0 (required); Type "LC" of a delay model
%                only
%     Floquet    struct(NHist = h, Count = c) with integers of at least 1: the
%                history grid and the number of leading Floquet multipliers
%                to report for a delay "LC" run (default none); Type "LC" of a
%                delay model only
%     Lyapunov   true adds the first Lyapunov coefficient and the GH label to
%                a delay "H" curve (logical scalar, default false); Type "H"
%                of a delay model only
%     Coefficients  true adds the zero-Hopf and Hopf-Hopf coefficients to the
%                ZH and HH specials of a delay "H" curve, as the fields of
%                Data.values (logical scalar, default false); Type "H" of a
%                delay model only
%
%   Options that are not given keep the engine defaults. If START is a
%   special point (any label except "EP"), the run does not report that
%   point again at its first step.
arguments
    m (1,1) phasecont.Model
    start
    options.Type = "EP"
    options.Free = []
    options.NTST = []
    options.NCOL = []
    options.Index = []
    options.MaxSteps = []
    options.Direction = []
    options.Bounds = []
    options.H0 = []
    options.Hmin = []
    options.Hmax = []
    options.TolX = []
    options.TolF = []
    options.MaxNewton = []
    options.Detect = missing
    options.Switch = []
    options.Plot (1,1) logical = false
    options.Var = []
    options.Params = []
    options.InclinationFlips = []
    options.MeshAdapt = []
    options.Roots = []
    options.Omega = missing
    options.Lyapunov = missing
    options.Coefficients = missing
    options.Eps = missing
    options.Floquet = missing
end
import phasecont.internal.num2json

type = string(options.Type);
if ~isscalar(type) || ~ismember(type, ["EP" "LP" "H" "LC" "LPC" "PD" "NS" "HOM"])
    error("phasecont:model:badOption", ...
        "Type must be ""EP"", ""LP"", ""H"", ""LC"", ""LPC"", ""PD"", ""NS"" or ""HOM"".");
end
isDelay = m.IsDelay;
if isDelay && type ~= "EP" && type ~= "H" && type ~= "LC"
    error("phasecont:model:badOption", ...
        "Type must be ""EP"", ""H"" or ""LC"" for a delay model, got ""%s"".", type);
end
isDelayHopf = isDelay && type == "H";
isDelayCycle = isDelay && type == "LC";
if isDelayCycle && ~(isempty(options.Roots) && ~isstruct(options.Roots))
    error("phasecont:model:badOption", ...
        "Roots does not apply to Type = ""LC"" of a delay model.");
end
roots = delayRoots(options.Roots, isDelay, isDelayHopf || isDelayCycle);
for name = ["Lyapunov" "Coefficients"]
    if ~isDelayHopf && ~isa(options.(name), "missing")
        error("phasecont:model:badOption", ...
            "%s applies to Type = ""H"" of a delay model only.", name);
    end
end
if ~(isDelayHopf || isDelayCycle) && ~isa(options.Omega, "missing")
    error("phasecont:model:badOption", ...
        "Omega applies to Type = ""H"" or ""LC"" of a delay model only.");
end
for name = ["Eps" "Floquet"]
    if ~isDelayCycle && ~isa(options.(name), "missing")
        error("phasecont:model:badOption", ...
            "%s applies to Type = ""LC"" of a delay model only.", name);
    end
end
if isDelay
    % The engine's delay entry follows equilibria, the Hopf curve through one
    % of their Hopf points and the cycles that start at one. A Hopf point
    % comes from the root count, and the other cycle options do not exist.
    if ~isempty(options.MeshAdapt)
        error("phasecont:model:badOption", ...
            "MeshAdapt does not apply to a delay model.");
    end
    if ~isempty(options.Switch)
        error("phasecont:model:badOption", ...
            "Switch does not apply to a delay model.");
    end
    if isDelayCycle && ~isempty(options.Var)
        error("phasecont:model:badOption", ...
            "Var does not apply to Type = ""LC"" of a delay model.");
    end
    if type == "EP" && ~isa(options.Detect, "missing") ...
            && ~all(ismember(string(options.Detect), ["LP" "BP"]))
        error("phasecont:model:badOption", ...
            "Detect for a delay model takes ""LP"" and ""BP"" only; a Hopf " + ...
            "point comes from the root count.");
    end
end
lyapunov = false;
coefficients = false;
omega = [];
if isDelayHopf
    if ~isa(options.Coefficients, "missing")
        coefficients = options.Coefficients;
        if ~(islogical(coefficients) && isscalar(coefficients))
            error("phasecont:model:badOption", ...
                "Coefficients must be a logical scalar.");
        end
    end
    if ~isa(options.Lyapunov, "missing")
        lyapunov = options.Lyapunov;
        if ~(islogical(lyapunov) && isscalar(lyapunov))
            error("phasecont:model:badOption", ...
                "Lyapunov must be a logical scalar.");
        end
    end
end
if (isDelayHopf || isDelayCycle) && ~isa(options.Omega, "missing")
    omega = options.Omega;
    if ~(isnumeric(omega) && isreal(omega) && isscalar(omega) && isfinite(omega))
        error("phasecont:model:badOption", ...
            "Omega must be a finite real scalar.");
    end
    omega = double(omega);
end
eps0 = [];
floquet = [];
if isDelayCycle
    eps0 = delayCycleEps(options.Eps);
    if ~isa(options.Floquet, "missing")
        floquet = delayCycleFloquet(options.Floquet);
    end
end
isCycle = type == "LC";
% A curve of cycles has two free parameters like an LP or H curve.
isCycleCurve = type == "LPC" || type == "PD" || type == "NS";
isHom = type == "HOM";
isCurve = type == "LP" || type == "H" || isCycleCurve || isHom;
flips = false;
if ~isempty(options.InclinationFlips)
    if ~isHom
        error("phasecont:model:badOption", ...
            "InclinationFlips applies to Type = ""HOM"" only.");
    end
    flips = options.InclinationFlips;
    if ~(islogical(flips) && isscalar(flips))
        error("phasecont:model:badOption", ...
            "InclinationFlips must be a logical scalar.");
    end
end
meshAdapt = options.MeshAdapt;
if ~isempty(meshAdapt)
    if ~(isCycle || isCycleCurve || isHom)
        error("phasecont:model:badOption", ...
            "MeshAdapt applies to Type = ""LC"", ""LPC"", ""PD"", ""NS"" or ""HOM"" only.");
    end
    if ~(isnumeric(meshAdapt) && isreal(meshAdapt) && isscalar(meshAdapt) ...
            && isfinite(meshAdapt) && meshAdapt >= 0 && meshAdapt == round(meshAdapt))
        error("phasecont:model:badOption", ...
            "MeshAdapt must be an integer of at least 0.");
    end
end
grid = [];
if isCycle
    if ~isa(options.Detect, "missing") || ~isempty(options.Switch)
        error("phasecont:model:badOption", ...
            "Detect and Switch do not apply to Type = ""LC"".");
    end
    grid = [countOption(options.NTST, "NTST", type), ...
        countOption(options.NCOL, "NCOL", type)];
elseif ~isCycleCurve && ~isHom && (~isempty(options.NTST) || ~isempty(options.NCOL))
    error("phasecont:model:badOption", ...
        "NTST and NCOL apply to Type = ""LC"", ""LPC"", ""PD"", ""NS"" and ""HOM"" only.");
end
if ~isHom && ~isempty(options.Index)
    error("phasecont:model:badOption", "Index applies to Type = ""HOM"" only.");
end
if isempty(options.Free)
    error("phasecont:model:badOption", "The Free option is required.");
end
free = string(options.Free);
if isCurve
    if numel(free) ~= 2 || free(1) == free(2)
        error("phasecont:model:badOption", ...
            "Free must name two different parameters for Type = ""%s"".", type);
    end
    free = reshape(free, 1, 2);
    if ~all(ismember(free, m.Parameters))
        error("phasecont:model:unknownParameter", ...
            "Free must be two of: %s.", strjoin(m.Parameters, ", "));
    end
    if ~isa(options.Detect, "missing") || ~isempty(options.Switch) ...
            || ~isempty(options.Var)
        error("phasecont:model:badOption", ...
            "Detect, Switch and Var do not apply to Type = ""%s"".", type);
    end
elseif ~isscalar(free) || ~any(m.Parameters == free)
    error("phasecont:model:unknownParameter", ...
        "Free must be one of: %s.", strjoin(m.Parameters, ", "));
end
if isDelayCycle
    [u0, p, ~] = startValues(m, start, options.Params);
    omega = delayHopfOmega(start, omega, "LC");
    % The third grid value is the number of Floquet multipliers per row, so
    % rows2branch checks the row width against it and does not infer it.
    grid(3) = 0;
    if ~isempty(floquet)
        grid(3) = floquet.Count;
    end
    from = "";
    mesh = [];
elseif isCycle
    [u0, p, from, mesh] = cycleStart(m, start, options.Params);
elseif isCycleCurve
    [u0, p, mesh, grid] = curveCycleStart(m, start, options.Params, type);
    if ~isempty(options.NTST)
        grid(1) = countOption(options.NTST, "NTST", type);
    end
    if ~isempty(options.NCOL)
        grid(2) = countOption(options.NCOL, "NCOL", type);
    end
elseif isHom
    [u0, p, from, mesh, grid] = homStart(m, start, options);
else
    [u0, p, special] = startValues(m, start, options.Params);
    if isDelayHopf
        omega = delayHopfOmega(start, omega, "H");
    elseif isDelay && special
        error("phasecont:model:badStart", ...
            "A delay run starts from an equilibrium (label ""EP""), not from ""%s"".", ...
            start.Label);
    end
end

parts = strings(1, 0);
parts(end+1) = numberOption("h0", options.H0, "H0");
parts(end+1) = numberOption("hmin", options.Hmin, "Hmin");
parts(end+1) = numberOption("hmax", options.Hmax, "Hmax");
parts(end+1) = numberOption("max_steps", options.MaxSteps, "MaxSteps", true);
parts(end+1) = numberOption("tol_x", options.TolX, "TolX");
parts(end+1) = numberOption("tol_f", options.TolF, "TolF");
parts(end+1) = numberOption("max_newton", options.MaxNewton, "MaxNewton", true);
if ~isempty(meshAdapt)
    parts(end+1) = """mesh_adapt"":" + num2json(double(meshAdapt));
end
if ~isempty(options.Direction)
    d = options.Direction;
    if ~(isnumeric(d) && isscalar(d) && (d == 1 || d == -1))
        error("phasecont:model:badOption", "Direction must be 1 or -1.");
    end
    parts(end+1) = """direction"":" + num2json(d);
end
if ~isempty(options.Bounds)
    b = options.Bounds;
    if ~(isnumeric(b) && isreal(b) && numel(b) == 2 && all(isfinite(b)) ...
            && b(1) < b(2))
        error("phasecont:model:badOption", ...
            "Bounds must be [lo hi] with finite values and lo < hi.");
    end
    parts(end+1) = """bounds"":[" + num2json(b(1)) + "," + num2json(b(2)) + "]";
end
if type == "EP" && isDelay
    if isa(options.Detect, "missing")
        detect = ["LP" "BP"];
    else
        detect = string(options.Detect);
    end
    parts(end+1) = """detect"":{""lp"":" + lower(string(any(detect == "LP"))) + ...
        ",""bp"":" + lower(string(any(detect == "BP"))) + "}";
elseif type == "EP"
    if isa(options.Detect, "missing")
        detect = ["LP" "BP" "H"];
    else
        detect = string(options.Detect);
    end
    if ~all(ismember(detect, ["LP" "BP" "H"]))
        error("phasecont:model:badOption", ...
            "Detect entries must be among ""LP"", ""BP"" and ""H"".");
    end
    parts(end+1) = """detect"":{""lp"":" + lower(string(any(detect == "LP"))) + ...
        ",""bp"":" + lower(string(any(detect == "BP"))) + ...
        ",""h"":" + lower(string(any(detect == "H"))) + "}";
end
parts = parts(strlength(parts) > 0);

if isHom
    problem = "{""type"":""HOM"",""free"":" + jsonencode(free) + ...
        ",""ntst"":" + num2json(grid(1)) + ",""ncol"":" + num2json(grid(2)) + ...
        ",""start"":""" + from + """";
    if from == "cycle"
        problem = problem + ",""mesh"":[" + ...
            strjoin(arrayfun(@num2json, mesh), ",") + "]";
    end
    if flips
        problem = problem + ",""inclination_flips"":true";
    end
    problem = problem + ",""options"":{" + strjoin(parts, ",") + "}";
elseif isCycleCurve
    problem = "{""type"":""" + type + """,""free"":" + jsonencode(free) + ...
        ",""ntst"":" + num2json(grid(1)) + ",""ncol"":" + num2json(grid(2)) + ...
        ",""mesh"":[" + strjoin(arrayfun(@num2json, mesh), ",") + "]" + ...
        ",""options"":{" + strjoin(parts, ",") + "}";
elseif isDelayCycle
    problem = "{""type"":""LC"",""free"":" + jsonencode(free) + ...
        ",""ntst"":" + num2json(grid(1)) + ",""ncol"":" + num2json(grid(2)) + ...
        ",""omega"":" + num2json(omega) + ",""eps"":" + num2json(eps0);
    if ~isempty(floquet)
        problem = problem + ",""floquet"":{""nhist"":" + num2json(floquet.NHist) + ...
            ",""count"":" + num2json(floquet.Count) + "}";
    end
    problem = problem + ",""options"":{" + strjoin(parts, ",") + "}";
elseif isDelayHopf
    problem = "{""type"":""H"",""free"":" + jsonencode(free) + ...
        ",""omega"":" + num2json(omega) + ...
        ",""lyapunov"":" + lower(string(lyapunov)) + ...
        ",""coefficients"":" + lower(string(coefficients));
    if roots.N > 0
        problem = problem + ",""roots"":{""n"":" + num2json(roots.N) + ...
            ",""re_min"":" + num2json(roots.ReMin) + "}";
    end
    problem = problem + ",""options"":{" + strjoin(parts, ",") + "}";
elseif isCurve
    problem = "{""type"":""" + type + """,""free"":" + jsonencode(free) + ...
        ",""options"":{" + strjoin(parts, ",") + "}";
elseif isCycle
    problem = "{""type"":""LC"",""free"":" + jsonencode(free) + ...
        ",""ntst"":" + num2json(grid(1)) + ",""ncol"":" + num2json(grid(2)) + ...
        ",""start"":""" + from + """";
    if from == "pd"
        problem = problem + ",""mesh"":[" + ...
            strjoin(arrayfun(@num2json, mesh), ",") + "]";
    end
    problem = problem + ",""options"":{" + strjoin(parts, ",") + "}";
elseif isDelay
    problem = "{""type"":""EP"",""free"":" + jsonencode(free) + ...
        ",""roots"":{""n"":" + num2json(roots.N) + ",""re_min"":" + ...
        num2json(roots.ReMin) + "},""options"":{" + strjoin(parts, ",") + "}";
else
    problem = "{""type"":""EP"",""free"":" + jsonencode(free) + ...
        ",""start_is_special"":" + lower(string(special)) + ...
        ",""options"":{" + strjoin(parts, ",") + "}";
end
hasSwitch = ~isempty(options.Switch);
if hasSwitch
    sw = string(options.Switch);
    if ~isscalar(sw)
        error("phasecont:model:badOption", "Switch must be one string.");
    end
    problem = problem + ",""switch"":" + jsonencode(sw);
end
problem = problem + "}";

varIdx = 0;
if ~isCurve && (options.Plot || ~isempty(options.Var))
    if isempty(options.Var)
        varIdx = 1;
    else
        v = options.Var;
        if isnumeric(v) && isscalar(v) && v >= 1 && v <= numel(m.Variables) ...
                && v == round(v)
            varIdx = double(v);
        elseif (isstring(v) || ischar(v)) && isscalar(string(v))
            varIdx = find(m.Variables == string(v), 1);
        else
            varIdx = [];
        end
        if isempty(varIdx)
            error("phasecont:model:unknownVariable", ...
                "Var must be one of: %s.", strjoin(m.Variables, ", "));
        end
    end
end
ax = [];
if options.Plot
    ax = newplot;
    if isCurve
        xlabel(ax, free(1));
        ylabel(ax, free(2));
    else
        xlabel(ax, free);
        ylabel(ax, m.Variables(varIdx));
    end
end

n = numel(m.Variables);
% Live plot rows: a curve draws a2 against a1, which sits after the n states,
% after the orbit and the period on a curve of cycles, or after the orbit,
% T and the saddle on a HOM curve; a cycle draws the maximum of the
% variable over its base points.
xrow = 1;
yrow = 1 + varIdx;
if isHom
    xrow = 3 + n*(grid(1)*grid(2) + 1) + n;
    yrow = 1;
elseif isCycleCurve
    xrow = 3 + n*(grid(1)*grid(2) + 1);
    yrow = 1;
elseif isCurve
    xrow = n + 2;
    yrow = 1;
elseif isCycle
    yrow = 1 + varIdx + n*(0:grid(1)*grid(2));
end
lines = gobjects(1, 0);
if options.Plot
    out = phasecont.internal.runLoop(m, problem, u0, p, @drawLive);
else
    out = phasecont.internal.runLoop(m, problem, u0, p);
end
st = out.Status;
nb = numel(st.branches);
br = phasecont.Branch.empty(1, 0);
record = phasecont.internal.RunRecord(struct( ...
    "Kind", "continuation", "Problem", problem, "U0", u0(:), "P", p(:), ...
    "ModelSha", m.SourceSha256, "Type", type, "Free", free, "Grid", reshape(grid, 1, []), ...
    "N", n, "Json", out.Json, "Points", {out.Points}, ...
    "Specials", {out.Specials}, "Stride", out.Stride));
for b = 1:nb
    one = phasecont.internal.rows2branch(m, free, p, n, out.Points{b}, ...
        out.Specials{b}, st.branches(b), type, grid);
    one.attachRecord(record, b);
    br(1, b) = one;
end
if options.Plot
    % Replace the live lines with the styled plot.
    delete(lines(isgraphics(lines)));
    for b = 1:nb
        if isCurve
            plot(br(b), Parent = ax);
        else
            plot(br(b), Var = varIdx, Parent = ax);
        end
    end
    drawnow
end

    function drawLive(pts, ~)
        % Draws the rows copied so far, one line per branch.
        for k = 1:numel(pts)
            if k > numel(lines)
                lines(k) = line(ax, NaN, NaN);
            end
            set(lines(k), XData = pts{k}(xrow, :), ...
                YData = max(pts{k}(yrow, :), [], 1));
        end
        drawnow limitrate
    end
end

function roots = delayRoots(value, isDelay, optional)
% The Roots option as a struct with N and ReMin. A delay model needs it, except
% for a Hopf curve where it only switches the Hopf-Hopf label on (N = 0 then
% means not given), and an ODE model must not have it. N is the degree of the pseudospectral
% interpolation that finds the characteristic roots, and the search keeps the
% roots with real part of at least ReMin, which must be negative so that every
% root in the right half plane is in range. Neither has a default.
roots = struct("N", 0, "ReMin", 0);
if isempty(value) && ~isstruct(value)
    if isDelay && ~optional
        error("phasecont:model:badOption", ...
            "A delay model needs the Roots option, for example Roots = struct(N = 30, ReMin = -0.5).");
    end
    return
end
if ~isDelay
    error("phasecont:model:badOption", ...
        "Roots applies to a delay model only.");
end
ok = isstruct(value) && isscalar(value) ...
    && isequal(sort(string(fieldnames(value))), ["N"; "ReMin"]);
if ok
    n = value.N;
    r = value.ReMin;
    ok = isnumeric(n) && isreal(n) && isscalar(n) && isfinite(n) ...
        && n >= 1 && n == round(n) ...
        && isnumeric(r) && isreal(r) && isscalar(r) && isfinite(r) && r < 0;
end
if ~ok
    error("phasecont:model:badOption", ...
        "Roots must be struct(N = n, ReMin = r) with an integer N of at least 1 and a ReMin below 0.");
end
roots = struct("N", double(value.N), "ReMin", double(value.ReMin));
end

function omega = delayHopfOmega(start, omega, kind)
% The frequency of a delay Hopf curve or of the cycles that start at a Hopf
% point. A Point start must carry an H label and supplies it in Data.omega, so
% Omega is then an error; a state vector needs it.
if isa(start, "phasecont.Point")
    if start.Label ~= "H"
        error("phasecont:model:badStart", ...
            "A delay ""%s"" run starts from an H point, not from ""%s"".", ...
            kind, start.Label);
    end
    if ~isempty(omega)
        error("phasecont:model:badOption", ...
            "Omega is taken from the H point; do not give it with a Point start.");
    end
    if ~(isstruct(start.Data) && isfield(start.Data, "omega") ...
            && isnumeric(start.Data.omega) && isscalar(start.Data.omega))
        error("phasecont:model:badStart", ...
            "The H point has no frequency in Data.omega.");
    end
    omega = double(start.Data.omega);
elseif isempty(omega)
    error("phasecont:model:badOption", ...
        "A state vector start needs the Omega option, the frequency of the Hopf point.");
end
end

function eps0 = delayCycleEps(value)
% The amplitude of the first orbit of a delay cycle run. It has no default.
if isa(value, "missing")
    error("phasecont:model:badOption", ...
        "The Eps option is required for Type = ""LC"" of a delay model.");
end
if ~(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) ...
        && value > 0)
    error("phasecont:model:badOption", ...
        "Eps must be a finite real scalar above 0.");
end
eps0 = double(value);
end

function fl = delayCycleFloquet(value)
% The Floquet option of a delay cycle run: struct(NHist = h, Count = c).
ok = isstruct(value) && isscalar(value) ...
    && isequal(sort(string(fieldnames(value))), ["Count"; "NHist"]);
if ok
    h = value.NHist;
    c = value.Count;
    ok = isnumeric(h) && isreal(h) && isscalar(h) && isfinite(h) ...
        && h >= 1 && h == round(h) ...
        && isnumeric(c) && isreal(c) && isscalar(c) && isfinite(c) ...
        && c >= 1 && c == round(c);
end
if ~ok
    error("phasecont:model:badOption", ...
        "Floquet must be struct(NHist = h, Count = c) with integers h and c of at least 1.");
end
fl = struct("NHist", double(value.NHist), "Count", double(value.Count));
end

function [u0, p, special] = startValues(m, start, params)
% Start state, parameters and whether the start is a special point.
n = numel(m.Variables);
np = numel(m.Parameters);
if isa(start, "phasecont.Point")
    if ~isscalar(start)
        error("phasecont:model:badStart", "START must be one Point.");
    end
    if ~isempty(start.Model) && start.Model ~= m
        error("phasecont:model:badStart", ...
            "The start point belongs to a different model.");
    end
    u0 = start.U;
    p = start.P;
    special = start.Label ~= "EP";
    if numel(u0) ~= n || numel(p) ~= np
        error("phasecont:model:badStart", ...
            "The start point needs %d states and %d parameters.", n, np);
    end
    p = overrideParams(m, p, params);
elseif isnumeric(start) && isreal(start) && isvector(start)
    if numel(start) ~= n
        error("phasecont:model:badStart", ...
            "The start vector needs %d states, got %d.", n, numel(start));
    end
    u0 = double(start(:));
    special = false;
    if isempty(params) && ~isstruct(params)
        p = m.ParameterValues;
    else
        p = m.paramVector(params);
    end
else
    error("phasecont:model:badStart", ...
        "START must be a phasecont.Point or a real state vector.");
end
end

function [u0, p, from, mesh] = cycleStart(m, start, params)
% Start values of an LC run. A point of a cycle branch (its Data has the
% period) must be a PD point; anything else goes to startValues and is
% taken as a Hopf point, which the engine checks.
mesh = [];
if isa(start, "phasecont.Point") && isscalar(start) && isstruct(start.Data) ...
        && isfield(start.Data, "period")
    if start.Label ~= "PD"
        error("phasecont:model:badStart", ...
            "An LC run starts from a Hopf point or a PD point, not from ""%s"".", ...
            start.Label);
    end
    if ~isempty(start.Model) && start.Model ~= m
        error("phasecont:model:badStart", ...
            "The start point belongs to a different model.");
    end
    if numel(start.P) ~= numel(m.Parameters)
        error("phasecont:model:badStart", ...
            "The start point needs %d parameters.", numel(m.Parameters));
    end
    u0 = [start.U(:); start.Data.period];
    p = overrideParams(m, start.P, params);
    mesh = double(start.Data.mesh(:)).';
    from = "pd";
else
    [u0, p] = startValues(m, start, params);
    from = "hopf";
end
end

function [u0, p, mesh, grid] = curveCycleStart(m, start, params, type)
% Start values of an LPC, PD or NS curve: a special point with the label
% TYPE on an LC branch, whose Data holds the period, the mesh, NTST and NCOL.
if ~(isa(start, "phasecont.Point") && isscalar(start) && isstruct(start.Data) ...
        && all(isfield(start.Data, ["period" "mesh" "ntst" "ncol"])) ...
        && start.Label == type)
    error("phasecont:model:badStart", ...
        "A %s curve starts from a %s point of an LC branch.", type, type);
end
if ~isempty(start.Model) && start.Model ~= m
    error("phasecont:model:badStart", ...
        "The start point belongs to a different model.");
end
if numel(start.P) ~= numel(m.Parameters)
    error("phasecont:model:badStart", ...
        "The start point needs %d parameters.", numel(m.Parameters));
end
u0 = [start.U(:); start.Data.period];
p = overrideParams(m, start.P, params);
mesh = double(start.Data.mesh(:)).';
grid = double([start.Data.ntst start.Data.ncol]);
end

function [u0, p, from, mesh, grid] = homStart(m, start, options)
% Start values of a HOM curve: point Index of an LC branch ("cycle"), or a
% struct with an orbit t, x and parameters p ("orbit").
n = numel(m.Variables);
if isa(start, "phasecont.Branch")
    if ~isscalar(start) || start.Type ~= "LC"
        error("phasecont:model:badStart", ...
            "A HOM curve starts from one LC Branch or from a struct with fields t, x and p.");
    end
    if ~isempty(start.Model) && start.Model ~= m
        error("phasecont:model:badStart", ...
            "The start branch belongs to a different model.");
    end
    i = options.Index;
    npts = numel(start.Period);
    if isempty(i)
        error("phasecont:model:badOption", ...
            "The Index option is required for Type = ""HOM"" from an LC branch.");
    end
    if ~(isnumeric(i) && isscalar(i) && isreal(i) && i == round(i) ...
            && i >= 1 && i <= npts)
        error("phasecont:model:badOption", ...
            "Index must be an integer from 1 to %d.", npts);
    end
    if numel(start.P) ~= numel(m.Parameters)
        error("phasecont:model:badStart", ...
            "The start branch needs %d parameters.", numel(m.Parameters));
    end
    u0 = [start.U(i, :).'; start.Period(i)];
    p = start.P;
    p(start.FreeIndex) = start.Alpha(i);
    p = overrideParams(m, p, options.Params);
    mesh = start.Mesh(i, :);
    grid = [start.NTST start.NCOL];
    if ~isempty(options.NTST)
        grid(1) = countOption(options.NTST, "NTST", "HOM");
    end
    if ~isempty(options.NCOL)
        grid(2) = countOption(options.NCOL, "NCOL", "HOM");
    end
    from = "cycle";
elseif isstruct(start) && isscalar(start) && all(isfield(start, ["t" "x" "p"]))
    if ~isempty(options.Index)
        error("phasecont:model:badOption", ...
            "Index applies to a HOM curve that starts from an LC branch.");
    end
    t = start.t;
    x = start.x;
    if ~(isnumeric(t) && isreal(t) && isvector(t) && numel(t) >= 2 ...
            && all(isfinite(t)) && all(diff(t(:)) > 0))
        error("phasecont:model:badStart", ...
            "The field t must hold at least two finite, increasing times.");
    end
    t = double(t(:));
    if ~(isnumeric(x) && isreal(x) && isequal(size(x), [numel(t) n]) ...
            && all(isfinite(x), "all"))
        error("phasecont:model:badStart", ...
            "The field x must be %d-by-%d, one row of finite states per time.", ...
            numel(t), n);
    end
    if isempty(start.p) && ~isstruct(start.p)
        error("phasecont:model:badStart", "The field p must hold the parameters.");
    end
    p = overrideParams(m, m.paramVector(start.p), options.Params);
    grid = [countOption(options.NTST, "NTST", "HOM from a struct"), ...
        countOption(options.NCOL, "NCOL", "HOM from a struct")];
    period = t(end) - t(1);
    % A uniform mesh: its base points are equally spaced over [0, 1].
    tau = (0:grid(1)*grid(2)).' / (grid(1)*grid(2));
    xs = interp1(t, double(x), t(1) + tau*period, "pchip");
    u0 = [reshape(xs.', [], 1); period];
    mesh = [];
    from = "orbit";
else
    error("phasecont:model:badStart", ...
        "A HOM curve starts from one LC Branch or from a struct with fields t, x and p.");
end
end

function p = overrideParams(m, p, params)
% P with the entries that PARAMS names replaced, or PARAMS as a whole vector.
if isstruct(params)
    v = m.paramVector(params);
    idx = find(ismember(m.Parameters, string(fieldnames(params))));
    p(idx) = v(idx);
elseif ~isempty(params)
    p = m.paramVector(params);
end
end

function v = countOption(value, name, type)
% A required positive integer option.
if isempty(value)
    error("phasecont:model:badOption", ...
        "The %s option is required for Type = ""%s"".", name, type);
end
if ~(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) ...
        && value >= 1 && value == round(value))
    error("phasecont:model:badOption", "%s must be a positive integer.", name);
end
v = double(value);
end

function s = numberOption(key, value, name, isCount)
% One "key":value entry, or "" when the option is not set.
if nargin < 4
    isCount = false;
end
s = "";
if isempty(value)
    return
end
if ~(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) ...
        && value > 0)
    error("phasecont:model:badOption", ...
        "%s must be a positive finite number.", name);
end
if isCount && value ~= round(value)
    error("phasecont:model:badOption", "%s must be an integer.", name);
end
s = """" + key + """:" + phasecont.internal.num2json(double(value));
end
