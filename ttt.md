Solution for PIP error in TRE

export PIP_CERT=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
pip install --cert /etc/ssl/certs/ca-certificates.crt openai
