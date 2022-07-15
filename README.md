###1. 애플리케이션 실행방법

```
$ cd python-fastapi

$ uvicorn app.api:app --port 8000

또는 

$ uvicorn app.main:app --port 8000
```

## 2. 도커 이미지 빌드 및 실행

```
$ cd python-fastapi

$ docker build -t <이미지 이름>:<이미지 버전>
$ docker build -t python-fastapi:0.0.1

$ docker run -d -p 8000:80 python-fastapi:0.0.1

```

## 3. ECR 에 이미지 Push

AWS ECR 에 접속하여 Repository 를 생성한다.
여기서는 예시로 python-fastapi 라는 Repository 를 생성했다고 가정함


레포지토리가 생성되어 있어야 ECR 에 이미지 Push 가 정상적으로 수행된다.
 

```
# AWS CLI 인증
wiki 참고

# AWS ECR 로그인
$ aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 423062414102.dkr.ecr.ap-northeast-2.amazonaws.com

$ docker tag <이미지 이름>:<이미지 버전> 423062414102.dkr.ecr.ap-northeast-2.amazonaws.com/<AWS ECR Repository 이름>:<이미지 버전>
$ docker tag python-fastapi:0.0.1 423062414102.dkr.ecr.ap-northeast-2.amazonaws.com/python-fastapi:0.0.1

$ docker push 423062414102.dkr.ecr.ap-northeast-2.amazonaws.com/<AWS ECR Repository 이름>:<이미지 버전>
$ docker push 423062414102.dkr.ecr.ap-northeast-2.amazonaws.com/python-fastapi:0.0.1

```

## 4. EKS Deployment, Service 배포
```
$ cd python-fastapi

$ kubectl apply -f deployment.yaml
$ kubectl apply -f service.yaml

```
