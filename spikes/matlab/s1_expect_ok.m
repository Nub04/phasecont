function ok = s1_expect_ok(label, f)
%S1_EXPECT_OK Calls f(), which must return without raising an error.
try
    f();
catch e
    ok = report(false, label, sprintf('%s | %s', e.identifier, e.message));
    return
end
ok = report(true, label, 'no error');
end
