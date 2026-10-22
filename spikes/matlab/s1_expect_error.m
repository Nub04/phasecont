function ok = s1_expect_error(label, f, id, texts)
%S1_EXPECT_ERROR Calls f(), which must raise error `id` whose message holds every string in texts.
%   Like expect_error, but texts may be a cell array and then all of them must appear.
texts = cellstr(texts);
try
    f();
catch e
    found = all(cellfun(@(t) contains(e.message, t), texts));
    ok = report(strcmp(e.identifier, id) && found, label, sprintf('%s | %s', e.identifier, e.message));
    return
end
ok = report(false, label, sprintf('no error, expected %s', id));
end
