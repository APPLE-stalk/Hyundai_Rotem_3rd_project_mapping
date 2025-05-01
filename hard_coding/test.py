data = [(1,2), (3,4), (1,2), (5,6), (3,4)]

print(type(data))
# 순서 상관없고 단순히 중복만 제거하고 싶을 때
unique = list(set(data))
print(unique)

print(type(unique))