#  Afinador e Metrônomo

## Descrição do problema / objetivo da aplicação

Músicos precisam, com frequência, de duas ferramentas básicas durante a prática: algo que mostre se o instrumento está afinado e algo que marque o andamento da música. 

Este projeto reúne as duas ferramentas em uma única aplicação, desenvolvida em **Python**. O **afinador** usa o microfone do computador para detectar a nota tocada em tempo real e informar o quanto ela está acima ou abaixo da afinação correta. O **metrônomo** gera cliques sonoros em um andamento (BPM) e compasso escolhidos pelo usuário.

## Principais funcionalidades

### Afinador
- Captura do áudio do microfone em segundo plano (thread de callback), sem travar a interface.
- Detecção da frequência fundamental usando o algoritmo **YIN**, com a diferença calculada via **FFT (Transformação Rápida de Fourier)** para ser mais rápida e refinada por interpolação parabólica.
- Conversão da frequência em **nota musical e oitava** (ex.: `A4`, `E2`).
- Cálculo do desvio em **cents** em relação à nota ideal.
- Indicador visual em escala de cores (🟥🟩🟨 ) com uma seta (▲) apontando a posição atual.
- Mensagens de orientação: **AFINADO** (tolerância de ±5 cents), "Muito baixo" ou "Muito alto".
- Retenção da última nota por 0,6 s para evitar que a tela fique piscando quando o som some.
- Botões para iniciar e parar o afinador.

### Metrônomo
- Ajuste do andamento de **30 a 240 BPM** por slider ou pelos botões **+** e **−**.
- Escolha do compasso: **2/4, 3/4, 4/4 e 6/8**.
- Som sintetizado por código (onda senoidal com decaimento): a primeira batida do compasso é mais aguda e forte que as demais.
- Botões de iniciar e parar, com indicação de status na tela.

### Geral
- Menu lateral para alternar entre **Afinador** e **Metrônomo**.
- Ao trocar de página, o áudio (microfone e metrônomo) é encerrado automaticamente.

## Tecnologias e bibliotecas utilizadas

| Tecnologia | Uso no projeto | Instalação |
|---|---|---|
| [Python 3.9+](https://www.python.org/) | Linguagem principal | [python.org/downloads](https://www.python.org/downloads/) |
| [Streamlit](https://streamlit.io/) | Interface web, sidebar, botões, slider e atualização periódica da tela com `st.fragment` | `pip install streamlit` |
| [NumPy](https://numpy.org/) | Processamento numérico, FFT e geração dos sons do metrônomo | `pip install numpy` |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Captura do microfone e reprodução do áudio do metrônomo | `pip install sounddevice` |

Bibliotecas da biblioteca padrão do Python (não precisam de instalação): `time` e `threading`.

## Instruções para instalação e execução

### Pré-requisitos

- [Python 3.9 ou superior](https://www.python.org/downloads/)
- `pip` (já vem com o Python)
- [Git](https://git-scm.com/)
- Microfone e saída de áudio (caixas ou fones) funcionando no computador

> ⚠️ A aplicação acessa o microfone e o alto-falante da **máquina onde ela está rodando** (via `sounddevice`). Por isso deve ser executada **localmente**. Hospedada em um servidor remoto, ela usaria o áudio do servidor, e não o do usuário.

### 1. Clonar o repositório

```bash
git clone https://github.com/TiagoMaianoo/streamlit.python.git
cd streamlit.python
```

### 2. (Recomendado) Criar e ativar um ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as bibliotecas externas

```bash
pip install streamlit numpy sounddevice
```

Ou, se preferir, crie um arquivo `requirements.txt` com o conteúdo abaixo e instale com `pip install -r requirements.txt`:

```text
streamlit
numpy
sounddevice
```

#### Dependência do sistema (somente Linux)

A `sounddevice` depende da biblioteca PortAudio. No Windows e no macOS ela já vem junto com o pacote do `pip`. No Linux (Debian/Ubuntu), instale antes:

```bash
sudo apt install libportaudio2
```

### 4. Executar a aplicação

```bash
streamlit run main.py
```

O navegador abrirá automaticamente em `http://localhost:8501`. Se isso não acontecer, acesse o endereço manualmente.

### 5. Como usar

1. No menu lateral, escolha **Afinador** ou **Metrônomo**.
2. **Afinador:** clique em *Iniciar afinador*, permita o acesso ao microfone se o sistema pedir e toque uma nota. Quando a seta estiver no bloco verde central, a nota está afinada.
3. **Metrônomo:** escolha o compasso, ajuste o BPM e clique em *▶ Iniciar*.

### Problemas comuns

- **"Não foi possível iniciar o microfone":** verifique se o microfone está conectado, se o sistema operacional liberou o acesso para o terminal/Python e se nenhum outro programa o está usando.
- **Sem som no metrônomo:** confira a saída de áudio padrão do sistema.
- **Erro com `st.fragment`:** atualize o Streamlit com `pip install --upgrade streamlit`.

## Estrutura do projeto

```
streamlit.python/
├── main.py    # Código da aplicação (afinador + metrônomo)
└── README.md
```

## Autor

Tiago de Lima Mariano |
