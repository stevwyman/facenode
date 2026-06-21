# facenode

This is my reference implementation of a face recognition service. Deployable
as micro-service to serve as "face detection engine" for my [RootNode project](https://github.com/stevwyman/rootnode).

The service is making use of the [deepface project](https://github.com/serengil/deepface).

Use the Dockerfile to build and the container-compose to deploy the service within
your own pod.

## Testing

You can access the service once deployed with a simple curl command sending
a picture.

```sh
curl -X POST -F "file=@IMG_3256.JPG" http://localhost:8001/detect
```
