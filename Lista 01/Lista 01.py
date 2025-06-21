from math import sqrt, log, ceil, erf, exp
from collections.abc import Callable
from csv import reader

def read_file(file_path: str) -> list[float]:
  with open(file_path, encoding = "utf-8") as file:
    return sorted(float(value) for row in reader(file) for value in row)

def mean(values: list[float]) -> float:
  if not values: return 0.0
  return sum(values) / len(values)

def mode(values: list[float]) -> float:
  if not values: return 0.0
  return max(set(values), key = values.count)

def median(values: list[float]) -> float:
  if not values: return 0.0
  div, mod = divmod(len(values), 2)
  return (values[div + mod - 1] + values[div]) / 2

def quartis(values: list[float]) -> tuple[float, float, float]:
  if not values: return 0.0, 0.0, 0.0
  div, mod = divmod(len(values), 2)
  q1 = median(values[:div])
  q2 = (values[div + mod - 1] + values[div]) / 2
  q3 = median(values[div + mod:])
  return (q1, q2, q3)

def amplitude(values: list[float]) -> float:
  if not values: return 0.0
  return max(values) - min(values)

def variance(values: list[float]) -> float:
  if not values: return 0.0
  mean_value = mean(values)
  sums = sum((value - mean_value) ** 2 for value in values)
  return sums / len(values)

def default_deviation(values: list[float]) -> float:
  if not values: return 0.0
  return sqrt(variance(values))

def variation_coefitient(values: list[float]) -> float:
  mean_value = mean(values)
  if not values or not mean_value: return 0.0
  return 100 * default_deviation(values) / mean_value

def detect_outliers(values: list[float], extreme: bool = False) -> list[float]:
  if not values: return []
  q1, _, q3 = quartis(values)
  _3_6_halfs_iqr = 1.5 * (q3 - q1) * (1 + extreme)
  lower, upper = q1 - _3_6_halfs_iqr, q3 + _3_6_halfs_iqr
  return [value for value in values if value < lower or value > upper]

def remove_outliers(values: list[float], extreme: bool = False) -> list[float]:
  if not values: return []
  q1, _, q3 = quartis(values)
  _3_6_halfs_iqr = 1.5 * (q3 - q1) * (1 + extreme)
  lower, upper = q1 - _3_6_halfs_iqr, q3 + _3_6_halfs_iqr
  return [value for value in values if lower <= value <= upper]

def histogram(values: list[float]) -> list[int]:
  if not values: return []
  bins = ceil(3.322 * log(len(values), 10) + 1)
  min_value, max_value = min(values), max(values)
  bin_size = (max_value - min_value) / bins
  histogram_data = [0] * bins
  for value in values:
    index = int((value - min_value) / bin_size)
    if index >= bins: index = bins - 1
    histogram_data[index] += 1
  return histogram_data

def plot_histogram(histogram_data: list[int]) -> None:
  if not histogram_data: return
  max_value = max(histogram_data)
  length = len(histogram_data)
  print(f"\nNúmero de classes: {length}")
  print("Histograma dos valores:")
  for i, count in enumerate(histogram_data):
    bar = "*" * (count * 50 // max_value)
    print(f"{i + 1}: {bar} (freq.: {count})")
  print("\n" + "=" * 50 + "\n")

def uniform(values: list[float]) -> Callable[[float], float]:
  if not values: return lambda x: 0.0
  diff = (b := max(values)) - (a := min(values))
  return lambda x: 0.0 if x < a else (x - a) / diff if x < b else 1.0

def exponential(values: list[float]) -> Callable[[float], float]:
  if not values: return lambda x: 0.0
  scale = len(values) / sum(values)
  return lambda x: 0.0 if x < 0 else 1 - exp(-scale * x)

def normal(values: list[float]) -> Callable[[float], float]:
  if not values: return lambda x: 0.0
  mu = sum(values) / (len_values := len(values))
  sigma = sqrt(2 * sum((v - mu) ** 2 for v in values) / len_values)
  return lambda x: 0.5 * (1 + erf((x - mu) / sigma))

def lognormal(values: list[float]) -> Callable[[float], float]:
  positive = [v for v in values if v > 0.0]
  if not values or not positive: return lambda x: 0.0
  mu = sum(log(v) for v in positive) / (len_positive := len(positive))
  sigma = sqrt(2 * sum((log(v) - mu) ** 2 for v in positive) / len_positive)
  return lambda x: 0.0 if x <= 0 else 0.5 * (1 + erf((log(x) - mu) / sigma))

def triangular(values: list[float]) -> Callable[[float], float]:
  if not values: return lambda x: 0.0
  a, b, c = min(values), mode(values), max(values)
  lower, upper = (c - a) * (b - a), (c - a) * (c - b)
  return lambda x: (0.0 if x < a else
    ((x - a) ** 2) / lower if x < b else
    1 - ((c - x) ** 2) / upper if x < c else 1.0)

KSFunction = Callable[[list[float]], Callable[[float], float]]

def kolmogorov_smirnov(values: list[float], distribution: KSFunction) -> bool:
  if not values: return False
  distrib_name = distribution.__name__.capitalize()
  try:
    n = len(values)
    empirical = [(i + 1) / n for i in range(n)]
    function = distribution(values)
    teorically = [function(x) for x in values]
  except ZeroDivisionError:
    print(f"Erro: Divisão por zero na distribuição \"{distrib_name}\".")
    return False
  d = max(abs(e - t) for e, t in zip(empirical, teorically))
  critical_value = 1.36 / sqrt(n)
  print(f"D {distrib_name}: {d}, Valor crítico: {critical_value}")
  return d < critical_value

def main() -> None:
  file_path = "./Lista 01/entrada.txt"

  with_outliers = read_file(file_path)
  without_outliers = remove_outliers(with_outliers)

  for values, label in ((with_outliers, "com"), (without_outliers, "sem")):
    print(f"\nAnálise estatística dos valores {label} outliers:\n")
    mean_value = mean(values)
    print(f"A média dos valores é: {mean_value}")
    mode_value = mode(values)
    print(f"A moda dos valores é: {mode_value}")
    median_value = median(values)
    print(f"A mediana dos valores é: {median_value}")
    q1, q2, q3 = quartis(values)
    print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")
    amplitude_value = amplitude(values)
    print(f"A amplitude dos valores é: {amplitude_value}")
    variance_value = variance(values)
    print(f"A variância dos valores é: {variance_value}")
    deviation_value = default_deviation(values)
    print(f"O desvio padrão dos valores é: {deviation_value}")
    coef_var_value = variation_coefitient(values)
    print(f"O coef. de variação dos valores é: {coef_var_value}%")
    outliers_values = detect_outliers(values)
    print(f"Valores considerados outliers: {outliers_values}")
    plot_histogram(histogram(values))
    for function in (uniform, exponential, normal, lognormal, triangular):
      function_name = function.__name__.capitalize()
      ks_result = "É" if kolmogorov_smirnov(values, function) else "Não é"
      print(f"Teste da distribuição {function_name}: {ks_result} aderente.\n")

if __name__ == "__main__": main()