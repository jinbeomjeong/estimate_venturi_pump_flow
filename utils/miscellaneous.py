

def count_divisions_by_two(num):
    """
    주어진 숫자가 1 이하가 될 때까지 2로 나눈 횟수를 계산합니다.

    Args:
        num (float or int): 1보다 큰 숫자.

    Returns:
        int: 2로 나눈 횟수.
    """
    if num <= 2:
        return 0  # 입력값이 이미 1 이하이면 0을 반환

    count = 0  # 나눈 횟수를 저장할 변수

    # 숫자가 1보다 큰 동안 계속 반복
    while num > 2:
        num /= 2  # 숫자를 2로 나눔
        count += 1  # 횟수 1 증가

    return count