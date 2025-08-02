import requests

url = "https://project-smash-production.up.railway.app/api/v1/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer a74d6ff6c32803a5c7fb5d33808a3121fcaa6f092c92b2fad1a60104ed663061"
}
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}

resp = requests.post(url, headers=headers, json=payload)
print(resp.status_code, resp.text)
