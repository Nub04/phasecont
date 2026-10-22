function ok = expect_error(label, f, id, text)
% Calls f(), which must raise error `id` with `text` in its message.
try
    f();
catch e
    ok = report(strcmp(e.identifier, id) && contains(e.message, text), label, ...
        sprintf('%s | %s', e.identifier, e.message));
    return
end
ok = report(false, label, sprintf('no error, expected %s', id));
end
