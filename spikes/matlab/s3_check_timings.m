function ok = s3_check_timings(tag, timings, method)
%S3_CHECK_TIMINGS Prints the timing rows of one S3 method and checks their names and values.
%   sparse_lu has the rows first and repeat. The condensed methods also have
%   condense, reduced and backsub. Every row needs min <= median <= max and a median above 0.
%   A null number decodes to [] and fails the check.
if strcmp(method, 'sparse_lu')
    want = {'first', 'repeat'};
else
    want = {'first', 'repeat', 'condense', 'reduced', 'backsub'};
end
rows = s3_rows(timings);
got = cell(1, numel(rows));
good = true;
for k = 1:numel(rows)
    t = rows(k);
    got{k} = t.name;
    fprintf('    %-9s median %s us, min %s us, max %s us\n', t.name, s4_numstr(t.median_ns / 1e3, '%.1f'), ...
        s4_numstr(t.min_ns / 1e3, '%.1f'), s4_numstr(t.max_ns / 1e3, '%.1f'));
    good = good && ~isempty(t.median_ns) && ~isempty(t.min_ns) && ~isempty(t.max_ns) ...
        && t.median_ns > 0 && t.min_ns <= t.median_ns && t.median_ns <= t.max_ns;
end
ok = report(isequal(got, want), [tag ' timing rows'], strjoin(got, ', '));
ok = report(good, [tag ' timing values'], 'median > 0 and min <= median <= max') && ok;
end
