## 1. 아스키 이미지 만들기

양자화는 영상이 얼마나 섬세하게 색을 표현할 수 있는지 결정합니다.

이번 문제에서는 영상을 흑백으로 바꾸어보고 주어진 레벨에 맞게 양자화를 진행합니다. 그리고 명도에 따라 아스키 문자를 할당하여 콘솔로 영상을 출력할 수 있는 간단한 아스키 영상을 만들어봅시다.

아래 함수 정의를 참고하여, 문제를 해결합니다.

**함수 정의**
`img2ascii(img, L, ascii_string)`

- 파라미터
  - `img` : OpenCV로 읽은 이미지
  - `L` : 양자화 레벨, 2<=L<=162<=*L*<=16 사이의 정수
  - `ascii_string` : 실습에서 주어지는 명도를 대신 표현할 문자열
- 반환값 : numpy.chararray 타입의 문자로 표현된 이미지

```
print_ascii_img(ascii_img)
```

- 기능 : 파라미터로 받은 `ascii_img`를 출력해주는 함수
- 파라미터
  - `ascii_img` : 문자로 이루어진 numpy.chararray 타입 이미지
- 반환값 : 없음

## 지시사항

1. 예시 영상을 읽고 흑백 영상으로 바꿉니다.
2. 주어진 양자화 레벨 ‘L’에 맞게 양자화를 진행합니다.
3. 양자화된 명도 ‘i’를 주어진 문자의 인덱스에 해당하는 문자로 치환하세요.
4. 치환된 이미지를 `print()`로 출력하여 결과를 확인합니다.

### Tips!

- `img2ascii()` 함수에서 변수 `L`을 이용해 양자화 후 `ascii_string`의 인덱스에 해당하는 문자로 치환하면 됩니다.



## 2. 액자 맞추기

어떤 액자 퍼즐이 있습니다. 이 퍼즐은 2x2 조각으로 구성이 되어있습니다.
![image](https://cdn-api.elice.io/api-attachment/attachment/cd84a843aeb44f28a708794285f122d6/puzzle.jpg)

퍼즐 조각은 아래처럼 2x2 조각으로 나눠져 있고 각 조각의 순서는 아래와 같습니다.
![image](https://cdn-api.elice.io/api-attachment/attachment/57f51d70a75d49b899fd43ada31e7c55/image.png)

주어진 `piece_order` 리스트는 현재 액자의 조각이 맞춰져 있는 순서를 나타내는 리스트입니다.

예를 들어 `piece_order = [3, 2, 1, 0]` 이라면 아래와 같은 의미를 나타냅니다.

- 0번 조각이 위치해야 할 곳에 3번 조각이 있음
- 1번 조각이 위치해야 할 곳에 2번 조각이 있음
- 2번 조각이 위치해야 할 곳에 1번 조각이 있음
- 3번 조각이 위치해야 할 곳에 0번 조각이 있음

액자 퍼즐을 원래대로 맞춰주세요.

## 지시사항

1. 주어진 영상과 조각 순서를 보고 원본 영상으로 복원한 영상을 반환하는 함수를 구현하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/4d9046cd0376499b9c1d502bb0da52c2/puzzle_solved.jpg)

### Tips!

- 조각 순서에 따라 현재 위치와 해당 조각이 이동해야 할 위치로 나눠서 생각해보세요.
- 인덱스를 나눴다면 이제 인덱스를 입력받아 조각의 위치를 조정하고, 조정된 이미지를 반환하는 함수를 구현해서 문제를 해결할 수 있습니다.



## 3. 이미지의 명도 분포 알아보기

영상은 색의 밝기로 표현이 됩니다.
OpenCV를 사용하면 명도가 0~255 사이의 값들로 표현이 됩니다.

우리는 특정 명도에 얼마나 색이 분포해있는지 알고 싶습니다. 예를 들어 명도가 10인 화소는 영상에서 몇 개나 되는지 세어보고 개수가 전체 화소의 90%에 해당한다면 이 영상은 저조도에서 촬영된 영상이라는 추론을 할 수 있게 됩니다.

그래서 우리는 이번에 특정 명도에 해당하는 영상 화소 수를 세어 명도별 영상 화소 수를 그래프로 그려보려 합니다.

흰색 바탕의 **256 × 화소 수** 크기의 numpy.ndarray를 만들고 이 이 영상에 그래프를 그리는 함수를 작성해주세요.

![image](https://cdn-api.elice.io/api-attachment/attachment/4e9daac15a304c3482bd30f0b3e8597a/image.png)

## 지시사항

1. 입력된 영상의 히스토그램을 반환하는 함수를 완성하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/27abc35286884e228ac47c2fb704b12b/image.png)

### Tips!

- numpy.ones 함수로 원소값이 1인 행렬을 만든 뒤 255를 곱하면 원소값이 255가 되어 백색 바탕 이미지를 만들 수 있습니다.

- OpenCV의 `cv2.line()`함수를 이용하면 손 쉽게 라인을 그릴 수 있습니다.

- 전체 화소수는 image의

   

  ```
  shape
  ```

   

  변수를 이용해서 구할 수 있습니다.

   

  ```
  shape
  ```

  은 아래와 같기 때문에 튜플의 모든 원소값을 곱한 것이 이미지가 가지고 있는 전체 화소 수가 됩니다.

  ```
  shape = (row, column, channel)
  Copy
  ```

예를 들어 명도가 128인 화소는 전체 이미지 중에 70번 등장했을 때 아래 처럼 코드를 쓸 수 있습니다.
`python cv2.line(img, (total_pixel, 128), (total_pixel - 70, 128))`

- 우리가 아는 그래프는 영상의 좌측하단 모서리가 (0, 0) 입니다. 그러나 영상을 표현하는 행렬의 시작점은 좌측 상단이 (0, 0) 이 됨을 유의하세요.



## 4. 이진 영상 만들기

이진 영상(Binary image)이란 영상의 명도가 0 또는 255로만 표현되는 영상을 의미합니다.

아래 영상처럼 번호판, 문서 등등 전경과 배경을 확실하게 분리해주는 효과가 있기 때문에 광학문자인식(OCR) 기술에서 전처리로 많이 쓰입니다. 예를 들어 아래와 같이 이진화한다면, 숫자가 더욱 뚜렷하게 구분됩니다.
![image](https://cdn-api.elice.io/api-attachment/attachment/f525e78cfd614101b1867ddcda0255fe/image.png)

이때 명도를 0 또는 255로 가르는 기준을 임곗값(Threshold)라고 합니다. 명도가 임곗값 **이상**이면 **255**로, **미만**이면 **0**으로 변환됩니다.

주어진 실습 영상을 이진 영상으로 바꾸어보세요.

## 지시사항

1. 주어진 임곗값을 기준으로 이진 이미지를 반환하는 함수를 구현하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/2ad5b27e75234e54a7cf008646687ad2/image.png)

### Tips

- 앞서 화소를 변경했던 방법을 떠올리며 문제를 해결하세요.



## 5. 이진화 응용 : 컬러 영상에서 특정 색상영역만 추출하기

앞선 실습에서는 영상을 단순히 임곗값을 가지고 아래처럼 0 또는 255 명도로 사상하였습니다.
`img[i, j] = 255 if img[i, j] >= threshold else 0`

이를 응용해서 컬러 영상에서 특정 색의 영역대는 명도를 255로 만들고 영역 밖의 색은 명도를 0으로 표시하려고 합니다.

예를 들어 R 채널에서 100 ~ 200 구간에 있는 명도만 살리고 싶은 경우 아래와 같이 표현할 수 있습니다.
`img[i, j, 0] = 255 if img[i, j, 0] >= range_start and img[i, j, 0] <= range_end else 0`

임곗값으로 아래와 같이 주어질 때, 특정 색상영역을 추출하는 실습을 해봅니다.
입력 : `[50, 255, 10, 100, 200, 255]`

## 지시사항

1. 컬러 이미지에서 특정 색상영역을 추출한 결과를 반환하는 함수를 완성하세요.
2. 주어지는 입력 `[50, 255, 10, 100, 200, 255]`은 다음을 의미합니다.
   - R채널에서는 [50, 255] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
   - G채널에서는 [10, 200] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
   - B채널에서는 [200, 255] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
3. 주어지는 임계값에 따라 특정 색상영역을 추출한 이미지를 반환하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/081299925e96485a9be6fd464313dabd/answer.jpg)





## 6. 프리윗 필터

OpenCV에는 필터를 영상에 쉽게 적용해주는 함수가 있습니다.

```
cv2.filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None) -> dst
```

• `src` : 입력 영상
• `ddepth` : 출력 영상 데이터 타입. (e.g) `cv2.CV_8U`, `cv2.CV_32F`, `cv2.CV_64F`, -1을 지정하면 `src`와 같은 타입의 `dst` 영상을 생성합니다.
• `kernel` : 필터 마스크 행렬. 실수형.
• `anchor`: 고정점 위치. (-1, -1)이면 필터 중앙을 고정점으로 사용
• `delta` : 추가적으로 더할 값
• `borderType` : 가장자리 픽셀 확장 방식
• `dst` : 출력 영상

이 함수를 이용하여 프리윗 필터를 영상에 적용시켜 봅시다.

## 지시사항

1. `prewitt()` 함수를 완성하세요.
2. `dst_vertical_edge` 변수에 수직 프리윗 필터를 적용하세요.
3. `dst_horizontal_edge` 변수에 수평 프리윗 필터를 적용하세요.
4. `img`에 수직/수평 프리윗 필터를 적용시킨 두 이미지 합을 반환하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/6361abd0e1334ee2b6e7531dc57998ec/image.png)





## 7. 회선처리

앞장에서는 OpenCV가 제공하는 필터 함수를 이용해 프리윗 필터를 적용해보았습니다.

이번에는 회선처리 함수를 직접 구현하여 프리윗 필터를 적용해보는 연습을 해보겠습니다.

구현의 편리함을 위해 아래와 같은 5 × 5 이미지가 있을 때, 경계선은 회선처리가 어려우므로 경계선을 제외한 내부 화소만 회선처리 하도록 합니다.

**이미지의 경계가 아닌 픽셀**
![image](https://cdn-api.elice.io/api-attachment/attachment/f2b269f4bf68466cbf9cc4ff823ee3c2/image.png)

**경계를 제외한 회선처리**
![image](https://cdn-api.elice.io/api-attachment/attachment/5a3c3c69dca74057a3c4fb9cb5b50e43/image.png)

## 지시사항

1. 주어진 커널을 입력 img에 회선 처리한 이미지 dst를 반환하는 함수 `convolution2D()`를 완성하세요.

**출력 예시**
![image](https://cdn-api.elice.io/api-attachment/attachment/d4b38980411a4cb69ce216a9ff95e035/answer.jpg)

### Tips!

- 내부 화소만 순회하기 : 문제에서 회선처리를 쉽게 하기 위해 경계선 화소들은 제외를 합니다.

  따라서 화소 순회를 할 때, 필터의 중점 좌표만큼 시작과 끝 범위를 정해주면 내부 경계선을 제외한 내부 화소만 순회가 가능합니다.

  ```
  center_r = kernel.shape[0] // 2
  center_c = kernel.shape[1] // 2
  
  for i in range(center_r, img.shape[0] - center_r):
          for j in range(center_c, img.shape[1] = center_c):
  Copy
  ```

- 4중 `for` 반복문이 사용됩니다. 첫 번째와 두 번째 반복문은 영상의 화소 순회를 하는 데 쓰이고, 세 번째와 네 번째 반복문은 필터의 요소 순회를 하는 데 쓰입니다.