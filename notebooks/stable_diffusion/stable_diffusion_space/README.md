# Server:

## Prerequisite

Python3.7+

## Install dependencies

```
pip install --upgrade pip
pip install wheel
pip install -r requirements_server.txt
pip install uvicorn[standard]
pip install <poptorch>.whl (not required in debug mode)
```

## Run

```
uvicorn server:app --host <server hostname> --port <port>
```

## Debug mode

poptorch, Poplar, torch, IPUs not needed

```
DEBUG=true uvicorn server:app --host <server hostname> --port <port>
```

# App:

## Install dependencies

```
pip install --uopgrade pip
pip install wheel
pip install -r requirements_app.txt
```

## Run

```
IPU_BACKEND=<server hostname>:<port> python app.py
```

Access the app via http://localhost:7860/.
