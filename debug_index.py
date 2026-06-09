import insurechat.app as app
print('Index texts:', len(app.index.texts))
if app.index.texts:
    print('Sample text 0:', app.index.texts[0][:200])
else:
    print('No texts loaded')
print('Query copay results count:', len(app.index.query('copay', topk=3)))
