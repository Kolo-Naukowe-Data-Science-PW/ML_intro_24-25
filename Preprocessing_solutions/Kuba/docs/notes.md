# Szczegółowy Raport Techniczny: Budowa Potoku Przetwarzania Danych dla Przewidywania Niespłacania Kredytów

## 1. Wprowadzenie i Motywacja

Głównym celem projektu było stworzenie gotowego do produkcji potoku przetwarzania danych wykorzystującego bibliotekę scikit-learn do przewidywania przypadków niespłacania kredytów. Wybrałem to podejście ze względu na jego praktyczność i możliwość zastosowania w rzeczywistych systemach bankowych.

Problem niespłacania kredytów jest krytycznym zagadnieniem dla instytucji finansowych. Skuteczny model predykcyjny może znacząco zmniejszyć ryzyko udzielania kredytów osobom, które mogą mieć trudności ze spłatą, co przekłada się na wymierne korzyści finansowe.

Moim początkowym założeniem było stworzenie rozwiązania, które będzie nie tylko dokładne, ale również łatwe w utrzymaniu i skalowaniu w środowisku produkcyjnym.

## 2. Pozyskiwanie i Eksploracja Danych

### 2.1 Ładowanie Danych

Dane zostały wprowadzone do systemu poprzez umieszczenie pliku `Loan_Default.csv` w katalogu `data` projektu. Do wczytania danych wykorzystałem bibliotekę pandas:

```python
import pandas as pd
import os

data_path = os.path.join('data', 'Loan_Default.csv')
df = pd.read_csv(data_path)
```

### 2.2 Eksploracyjna Analiza Danych (EDA)

Przed przystąpieniem do budowy potoku przeprowadziłem szczegółową analizę danych:

```python
print(df.info())
print(df.describe())

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
```

Wygenerowałem również szereg wizualizacji, w tym:
- Histogramy rozkładów zmiennych numerycznych
- Wykresy słupkowe dla zmiennych kategorycznych
- Mapy korelacji między zmiennymi

W trakcie analizy napotkałem kilka wyzwań związanych z jakością danych:
- Brakujące wartości w kilku kluczowych kolumnach
- Skośne rozkłady niektórych zmiennych numerycznych
- Niezbalansowany rozkład klasy docelowej (niespłacone kredyty stanowiły mniejszość)

## 3. Przetwarzanie Wstępne Danych

### 3.1 Automatyczne Wykrywanie Typów Zmiennych

Zaimplementowałem mechanizm automatycznego wykrywania zmiennych numerycznych i kategorycznych:

```python
def identify_column_types(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_columns, categorical_columns
```

### 3.2 Obsługa Brakujących Wartości

Do uzupełnienia brakujących wartości wykorzystałem `SimpleImputer` z różnymi strategiami:

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

### 3.3 Skalowanie i Kodowanie

Zastosowałem standardowe techniki przekształcania danych:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore')
```

Wybór tych metod był podyktowany ich skutecznością w praktyce oraz dobrą integracją z potokiem scikit-learn. Głównym wyzwaniem było zapewnienie spójności transformacji między etapem trenowania a wnioskowania.

## 4. Niestandardowe Inżynieria Cech

Zaprojektowałem własne transformatory do tworzenia nowych cech, które mogłyby zwiększyć moc predykcyjną modelu:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class IncomeToLoanRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, income_col='Income', loan_col='LoanAmount'):
        self.income_col = income_col
        self.loan_col = loan_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        X_copy['IncomeToLoanRatio'] = X_copy[self.income_col] / X_copy[self.loan_col].replace(0, 0.01)
        return X_copy
```

Motywacją dla utworzenia tej cechy było moje założenie, że stosunek dochodu do kwoty kredytu jest istotnym wskaźnikiem zdolności kredytowej. Napotkałem jednak problem z wartościami zerowymi w mianowniku, który rozwiązałem przez zastąpienie ich małą wartością.

## 5. Integracja Modelu i Trenowanie

### 5.1 Budowa Potoku

Zintegrowałem wszystkie elementy przetwarzania danych i model predykcyjny w jednym potoku:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('custom_features', IncomeToLoanRatioTransformer()),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

### 5.2 Podział Danych i Trenowanie

Zastosowałem standardowy podział na zbiory treningowy i testowy:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

X = df.drop('LoanStatus', axis=1)
y = df['LoanStatus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

## 6. Zapisywanie i Ponowne Ładowanie Potoku

Do zapisania i późniejszego załadowania potoku wykorzystałem bibliotekę joblib:

```python
import joblib

joblib.dump(full_pipeline, 'loan_default_pipeline.joblib')

loaded_pipeline = joblib.load('loan_default_pipeline.joblib')

y_pred_loaded = loaded_pipeline.predict(X_test)
print(f"Accuracy (loaded model): {accuracy_score(y_test, y_pred_loaded)}")
```

Napotkałem problemy z instalacją joblib w moim środowisku Conda, które rozwiązałem przez ręczną instalację:

```bash
conda install -c anaconda joblib
```

## 7. Ulepszenia i Dodatkowe Analizy

### 7.1 Strojenie Hiperparametrów

Przeprowadziłem strojenie hiperparametrów za pomocą przeszukiwania siatki:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy wynik F1: {grid_search.best_score_}")
```

### 7.2 Radzenie Sobie z Niezbalansowaniem Klas

Zastosowałem technikę SMOTE do generowania syntetycznych przykładów klasy mniejszościowej:

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

imb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('custom_features', IncomeToLoanRatioTransformer()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

imb_pipeline.fit(X_train, y_train)
y_pred_imb = imb_pipeline.predict(X_test)
print(f"F1 score z SMOTE: {f1_score(y_test, y_pred_imb)}")
```

## 8. Wady, Wyzwania i Wnioski

### 8.1 Napotkane Problemy

Podczas realizacji projektu napotkałem szereg wyzwań:

1. **Niezgodność kolumn** - Miałem trudności z zapewnieniem spójności kolumn między etapem preprocessingu a etapem trenowania. Problem rozwiązałem przez dokładne projektowanie potoku.

2. **Parsowanie argumentów w środowisku notebooka** - Początkowo próbowałem wykorzystać argparse do parametryzacji mojego potoku, co okazało się problematyczne w środowisku Jupyter. Przeszedłem na wykorzystanie zmiennych konfiguracyjnych.

3. **Wycieki danych** - Musiałem zachować ostrożność, aby transformacje danych (np. skalowanie) były zawsze wykonywane wewnątrz walidacji krzyżowej, a nie przed nią.

### 8.2 Ograniczenia Obecnego Podejścia

Moje rozwiązanie ma kilka istotnych ograniczeń:

- Brak mechanizmu monitoringu dryfu danych w czasie rzeczywistym
- Ograniczona obsługa nietypowych wartości wejściowych
- Potrzeba regularnego ponownego trenowania w celu utrzymania aktualności modelu

### 8.3 Wnioski

Z doświadczeń wyniesionych z projektu ustaliłem kilka kluczowych zasad do przyszłych projektów:

- Zawsze projektować potoki danych z myślą o środowisku produkcyjnym
- Starannie testować zachowanie potoku dla różnych typów danych wejściowych
