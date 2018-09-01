# Sample Recommendation System API Application

Build Recommendation System and deploy using Tornado Web Framework.

## Setup Environment on Local Machine

### Installation

```
git clone <url>

cd <repo>  # cd recommendation-api

# Install Packages
python env/create_env.py
source activate env/venv  # Windows users: activate env/venv
python env/install_packages.py

# Build the Model
python ml_src/build_model.py

# Run the App
python run.py
````

### Test App


1. Open Browser:  [http://localhost:9000](http://localhost:9000)

2. Command Line:

```
curl -i http://localhost:9000/api/recommendation/predict -X POST -d '{ "activeUser": 2, "numberOfRec": 5}'
```

Returns:

```
HTTP/1.1 200 OK
Server: TornadoServer/5.0.2
Content-Type: text/html; charset=UTF-8
Date: Sat, 01 Sep 2018 08:42:14 GMT
Content-Length: 45

{"status": 200, "data": [64, 59, 12, 50, 89]}
```

The data in the JSON is the movieIds.

### Note: It contains pickeled P and Q matrices for only first 100 users. If you want to train the model and find recommendation for all the users, you can do:

```
cd ml_src/LatentCollaborativeFiltering.py
```

You can also increase the step size and tweak hyperparameters for your choice.

## Credits:



### The End.