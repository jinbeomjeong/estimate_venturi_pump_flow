import numpy as np
from scipy.interpolate import CubicSpline


def time_warp(series, sigma=0.2, knots=4):
    """
    시계열 데이터에 시간 축 왜곡을 적용합니다.

    Args:
        series (np.array): 1D 시계열 데이터
        sigma (float): 왜곡의 강도를 조절하는 표준편차
        knots (int): 왜곡을 위한 기준점(knot)의 개수

    Returns:
        np.array: 왜곡이 적용된 새로운 시계열 데이터
    """
    time_steps = len(series)

    # 1. 기준점(knot) 선택
    knot_x = np.linspace(0, time_steps - 1, knots)

    # 2. 기준점 왜곡 (랜덤 노이즈 추가)
    knot_y_random = np.random.normal(loc=1.0, scale=sigma, size=(knots,))

    # 3. 부드러운 곡선(Spline) 생성
    # 기준점과 랜덤 노이즈를 곱하여 y축을 왜곡
    spline_func = CubicSpline(knot_x, knot_x * knot_y_random)
    warped_time_index = spline_func(np.arange(time_steps))

    # 4. 데이터 재샘플링 (보간)
    # 원본 데이터의 인덱스와 값, 그리고 왜곡된 시간 인덱스를 사용
    warped_series = np.interp(warped_time_index, np.arange(time_steps), series)

    return warped_series


def count_divisions_by_two(number):
    """
    주어진 숫자를 2로 몇 번 나눌 수 있는지 계산합니다. (반복문 사용)

    Args:
        number (int): 확인할 정수.

    Returns:
        int: 2로 나눌 수 있는 횟수.
    """
    # 0 또는 음수는 처리하지 않으므로 0을 반환합니다.
    if number <= 0:
        return 0

    count = 0
    # 숫자가 2로 나누어떨어지는 동안 반복합니다.
    while number % 2 == 0:
        number //= 2  # number를 2로 나눈 몫으로 업데이트합니다.
        count += 1  # 횟수를 1 증가시킵니다.

    return count