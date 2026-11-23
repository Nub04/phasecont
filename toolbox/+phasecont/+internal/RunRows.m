classdef RunRows < handle
    % The rows of one engine run, copied as they arrive. Internal.
    %
    %   rows = phasecont.internal.RunRows(live)
    %   pts = rows.pull(run, st)
    %   out = rows.collect(run, st)
    %
    % Rows arrive in chunks. With live false the chunks are joined once in
    % collect, so a long run does not copy its whole matrix on every poll.
    % With live true the joined matrices are kept up to date after every
    % pull, for a caller that draws them while the run goes. runLoop and
    % runMany share this class.

    properties (Access = private)
        Live (1,1) logical = false
        Chunks = cell(1, 0)
        Have = zeros(1, 0)
        Pts = cell(1, 0)
    end

    methods
        function obj = RunRows(live)
            obj.Live = live;
        end

        function pts = pull(obj, run, st)
            % Copies the rows that arrived since the last pull. PTS is the
            % cell of joined rows when live is on, and empty otherwise.
            nb = numel(st.branches);
            for b = 1:nb
                if b > numel(obj.Chunks)
                    obj.Chunks{b} = {};
                    obj.Have(b) = 0;
                    obj.Pts{b} = zeros(st.stride, 0);
                end
                avail = st.branches(b).points;
                if avail > obj.Have(b)
                    block = run.copy(b, 0, obj.Have(b) + 1, ...
                        avail - obj.Have(b), st.stride);
                    obj.Have(b) = avail;
                    if obj.Live
                        obj.Pts{b} = [obj.Pts{b}, block];
                    else
                        obj.Chunks{b}{end+1} = block;
                    end
                end
            end
            if obj.Live
                pts = obj.Pts;
            else
                pts = cell(1, 0);
            end
        end

        function out = collect(obj, run, st)
            % The result of a run that has stopped, after its last pull: a
            % struct with Status, Json, Points, Specials and Stride (see
            % runLoop).
            nb = numel(st.branches);
            pts = obj.Pts;
            if ~obj.Live
                for b = 1:nb
                    pts{b} = [zeros(st.stride, 0), obj.Chunks{b}{:}];
                end
            end
            sp = cell(1, nb);
            for b = 1:nb
                nsp = numel(st.branches(b).specials);
                sp{b} = run.copy(b, 1, 1, nsp, st.stride);
            end
            out = struct("Status", st, "Json", run.LastStatus, ...
                "Points", {pts}, "Specials", {sp}, "Stride", st.stride);
        end
    end
end
