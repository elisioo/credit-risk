import pickle
import os
p = os.path.join(os.path.dirname(__file__), '..', 'streamlit-app-credit-risk', 'models', 'credit_risk_model.pkl')
try:
    with open(p, 'rb') as f:
        m = pickle.load(f)
    print('Loaded OK, type:', type(m))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('EXC_REPR:', repr(e))
