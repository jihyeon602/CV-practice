from tensorflow.keras import datasets, layers, models, activations


# 모델 변수를 선언합니다.
model = models.Sequential()

# 모델에 첫 번째 입력 레이어를 추가합니다.
# 컨볼루션 레이어 만들기, 커널사이즈 (3,3)
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))


# 아래에 지시상항에 있는 모델 구조가 되도록 나머지 모델 구조를 선언해주세요.
# 풀링 적용하기
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 1차원  텐서로 변환하기
model.add(layers.Flatten())  #1차원: 576

# FC레이어 만들기
model.add(layers.Dense(64, activation='relu'))  #특징추출-추상화
model.add(layers.Dense(10, activation='softmax')) #10개 클래스 중 하나로 최종 분류(softmax-확률분포로 변환)

# 모델 구조 출력하기
model.summary()
