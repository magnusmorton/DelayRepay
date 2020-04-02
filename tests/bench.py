import delayarray as m
data = m.random.random((1000, 1000))
print(m.sin(data) ** 2 + m.cos(data) ** 2)
