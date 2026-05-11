import random
import time

def create_character():
    print("=== 무인도 생존 시뮬레이션 ===")
    
    name: str = input("당신의 이름은 무엇인가요? ")
    weight = random.randint(45, 95)
    
    print(f"환영합니다, {name} 생존자님!")
    print(f"당신의 체중은 {weight}kg 으로 설정되었습니다.")
    
    hp = 80
    hunger = 86.7
    print(f"초기 상태 → 체력: {hp}, 배고픔: {hunger}")
    
    return name, weight, hp, hunger

def print_status(name, weight, hp, hunger):
    print("\n=== 생존자 상태 ===")
    print(f"이름: {name}")
    print(f"체중: {weight}kg")
    print(f"체력: {hp}")
    print(f"배고픔: {hunger}")

name, weight, hp, hunger = create_character()

print("주위를 둘러보는 중...")
time.sleep(2)
print("아무것도 발견하지 못했다.")

print_status(name, weight, hp, hunger)
