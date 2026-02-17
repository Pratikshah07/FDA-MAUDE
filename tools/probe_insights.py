import os, sys
sys.path.insert(0, os.getcwd())
from app import app
from io import BytesIO

p = app.test_client()
# Login using test account
login_payload = {'email': 'test@maude.local', 'password': 'Test12345'}
rv = p.post('/api/login', json=login_payload)
print('/api/login ->', rv.status_code, rv.get_data(as_text=True))

with open('tests/data/test_dashboard.csv','rb') as f:
    data = {'file': (BytesIO(f.read()), 'test_dashboard.csv'), 'process_raw': 'off'}
    rv = p.post('/insights', data=data, content_type='multipart/form-data')
    print('/insights ->', rv.status_code)
    b = rv.get_data(as_text=True)
    print('contains Plotly warning?', 'Plotly' in b)
    print('\n--- HTML SNIPPET ---\n')
    print(b[:2000])
