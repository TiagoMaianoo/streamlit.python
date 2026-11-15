import time
import threading

import numpy as np
import sounddevice as sd
import streamlit as st


# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================

TAXA_AMOSTRAGEM = 44100
TAMANHO_BLOCO = 4096

FREQUENCIA_MINIMA = 60
FREQUENCIA_MAXIMA = 1000

# Intervalo de atualização da tela do afinador (segundos).
INTERVALO_ATUALIZACAO = 0.25

# Tempo (segundos) que a última nota continua na tela
# depois que o som some, para evitar "piscar" entre nota e "--".
TEMPO_RETENCAO = 0.6

# Tolerância (em cents) para considerar a nota AFINADA.
TOLERANCIA_AFINADO = 5


# ==========================================
# CAPTURA DE ÁUDIO (EM SEGUNDO PLANO)
# ==========================================

class CapturaAudio:
    """
    Captura o microfone em uma thread de callback e mantém sempre
    os últimos TAMANHO_BLOCO samples. Assim, a tela nunca precisa
    esperar o microfone (stream.read bloqueava e causava o piscar).
    """

    def __init__(self):
        self.buffer = np.zeros(TAMANHO_BLOCO, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        dados = indata[:, 0]

        with self.lock:
            if len(dados) >= TAMANHO_BLOCO:
                self.buffer = dados[-TAMANHO_BLOCO:].copy()
            else:
                self.buffer = np.concatenate(
                    (self.buffer[len(dados):], dados)
                )

    def iniciar(self):
        if self.stream is not None:
            return

        self.buffer = np.zeros(TAMANHO_BLOCO, dtype=np.float32)

        self.stream = sd.InputStream(
            samplerate=TAXA_AMOSTRAGEM,
            channels=1,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()

    def parar(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    @property
    def ativo(self):
        return self.stream is not None

    def ultimo_bloco(self):
        with self.lock:
            return self.buffer.copy()


# ==========================================
# AFINADOR - DETECTAR FREQUÊNCIA (YIN COM FFT)
# ==========================================

def detectar_frequencia(audio):
    """
    Recebe um bloco de amostras de áudio e retorna a frequência
    fundamental estimada em Hz. Se não houver sinal válido, retorna None.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return None

    audio = audio - np.mean(audio)

    amplitude = np.max(np.abs(audio))

    if amplitude < 0.01:
        return None

    audio = audio / amplitude

    n = audio.size
    atraso_maximo = int(TAXA_AMOSTRAGEM / FREQUENCIA_MINIMA)
    atraso_minimo = int(TAXA_AMOSTRAGEM / FREQUENCIA_MAXIMA)

    janela = n - atraso_maximo

    if janela <= 0:
        return None

    # Diferença do YIN calculada com FFT (muito mais rápida que o laço):
    # d(t) = soma(x[j]^2) + soma(x[j+t]^2) - 2 * autocorrelação(t)
    tamanho_fft = 1
    while tamanho_fft < n + atraso_maximo:
        tamanho_fft *= 2

    fft_sinal = np.fft.rfft(audio, tamanho_fft)
    fft_janela = np.fft.rfft(audio[:janela], tamanho_fft)

    autocorrelacao = np.fft.irfft(
        fft_sinal * np.conj(fft_janela)
    )[: atraso_maximo + 1]

    energia = np.concatenate(([0.0], np.cumsum(audio ** 2)))

    atrasos = np.arange(atraso_maximo + 1)

    diferenca = (
        energia[janela]
        + (energia[atrasos + janela] - energia[atrasos])
        - 2 * autocorrelacao
    )
    diferenca[0] = 0.0

    # Normalização acumulada.
    normalizada = np.ones(atraso_maximo + 1)

    soma_acumulada = np.cumsum(diferenca[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        valores = diferenca[1:] * atrasos[1:] / soma_acumulada

    valores = np.where(soma_acumulada > 0, valores, 1.0)
    normalizada[1:] = valores

    # Procura o primeiro atraso que atende ao limite do YIN.
    atraso = None

    for i in range(atraso_minimo, atraso_maximo):
        if normalizada[i] < 0.15:
            atraso = i

            while (
                atraso + 1 <= atraso_maximo
                and normalizada[atraso + 1] < normalizada[atraso]
            ):
                atraso += 1

            break

    # Se nenhum ponto atingir o limite, escolhe o menor valor da região.
    if atraso is None:
        regiao = normalizada[atraso_minimo:atraso_maximo]

        if regiao.size == 0:
            return None

        atraso = int(np.argmin(regiao)) + atraso_minimo

    # Refinamento por interpolação parabólica.
    if 1 < atraso < atraso_maximo:
        anterior = normalizada[atraso - 1]
        atual = normalizada[atraso]
        posterior = normalizada[atraso + 1]

        denominador = anterior - 2 * atual + posterior

        if denominador != 0:
            ajuste = 0.5 * (anterior - posterior) / denominador
            atraso = atraso + ajuste

    if atraso <= 0:
        return None

    frequencia = TAXA_AMOSTRAGEM / atraso

    if not FREQUENCIA_MINIMA <= frequencia <= FREQUENCIA_MAXIMA:
        return None

    return float(frequencia)


# ==========================================
# AFINADOR - CONVERTER FREQUÊNCIA EM NOTA
# ==========================================

def analisar_nota(frequencia):
    """
    Recebe uma frequência em Hz e retorna:
    nome da nota, oitava, frequência-alvo e diferença em cents.
    """

    if frequencia is None or frequencia <= 0:
        return None

    notas = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    numero_nota = int(round(
        69 + 12 * np.log2(frequencia / 440)
    ))

    nota = notas[numero_nota % 12]
    oitava = (numero_nota // 12) - 1

    alvo = 440 * (2 ** ((numero_nota - 69) / 12))

    cents = 1200 * np.log2(frequencia / alvo)

    return nota, oitava, alvo, cents


# ==========================================
# AFINADOR - ESCALA DE CORES (EMOJIS)
# ==========================================

# 11 posições, de -50 a +50 cents (10 cents por bloco).
# O bloco central (índice 5) cobre -5 a +5 cents = AFINADO.
ESCALA_CORES = [
    "🟥", "🟥", "🟨", "🟨", "🟩", "🟩", "🟩", "🟨", "🟨", "🟥", "🟥"
]

POSICAO_CENTRAL = len(ESCALA_CORES) // 2  # 5

# Espaço de largura total (mesma largura de um emoji),
# usado para alinhar a seta com o bloco certo.
ESPACO_LARGO = "\u3000"


def posicao_na_escala(cents):
    """Converte cents (-50 a +50) em índice de 0 a 10."""
    cents = float(np.clip(cents, -50, 50))
    return int(round((cents + 50) / 10))


SETA = "▲"


def desenhar_escala(cents=None):
    """
    Desenha a escala em 11 colunas. Cada coluna tem o bloco colorido
    em cima e, embaixo, a seta (só na coluna ativa) ou um ponto discreto.
    """

    total = len(ESCALA_CORES)

    if cents is None:
        cores = ["⬜"] * total
        posicao = POSICAO_CENTRAL
        seta = "⚪"
    else:
        cores = ESCALA_CORES
        posicao = posicao_na_escala(cents)
        seta = SETA

    colunas = st.columns(total, gap="small")

    for i, coluna in enumerate(colunas):
        with coluna:
            st.markdown(cores[i])
            st.markdown(seta if i == posicao else "·")


# ==========================================
# CLASSE DO METRÔNOMO
# ==========================================

class Metronomo:

    def __init__(self):
        self.taxa = 44100

        self.bpm = 120
        self.tempos = 4

        self.lock = threading.Lock()

        self.stream = None

        self.amostras_ate_batida = 0
        self.posicao_som = 0

        self.som_atual = np.zeros(1, dtype=np.float32)

        self.batida_atual = 0

        self.som_normal = self.criar_som(1000)
        self.som_forte = self.criar_som(1600)

    # --------------------------------------
    # GERAR SONS
    # --------------------------------------

    def criar_som(self, frequencia):
        duracao = 0.04

        tempo = np.arange(int(self.taxa * duracao)) / self.taxa

        som = (
            np.sin(2 * np.pi * frequencia * tempo)
            * np.exp(-tempo * 80)
            * 0.6
        )

        return som.astype(np.float32)

    # --------------------------------------
    # ATUALIZAR CONFIGURAÇÕES
    # --------------------------------------

    def atualizar(self, bpm, tempos):
        with self.lock:
            self.bpm = bpm
            self.tempos = tempos

    # --------------------------------------
    # PROCESSAR ÁUDIO
    # --------------------------------------

    def executar_audio(self, outdata, frames, time_info, status):
        outdata.fill(0)

        for i in range(frames):
            if self.amostras_ate_batida <= 0:
                with self.lock:
                    bpm = self.bpm
                    tempos = self.tempos

                intervalo = int(self.taxa * 60 / bpm)

                self.amostras_ate_batida = intervalo

                # Primeira batida do compasso mais forte.
                if self.batida_atual == 0:
                    self.som_atual = self.som_forte
                else:
                    self.som_atual = self.som_normal

                self.batida_atual = (self.batida_atual + 1) % tempos

                self.posicao_som = 0

            # Reproduzir o som da batida.
            if self.posicao_som < len(self.som_atual):
                outdata[i, 0] = self.som_atual[self.posicao_som]
                self.posicao_som += 1

            self.amostras_ate_batida -= 1

    # --------------------------------------
    # INICIAR
    # --------------------------------------

    def iniciar(self, bpm, tempos):
        self.parar()

        self.atualizar(bpm, tempos)

        self.amostras_ate_batida = 0
        self.posicao_som = 0
        self.batida_atual = 0

        self.stream = sd.OutputStream(
            samplerate=self.taxa,
            channels=1,
            blocksize=512,
            dtype="float32",
            callback=self.executar_audio
        )

        self.stream.start()

    # --------------------------------------
    # PARAR
    # --------------------------------------

    def parar(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ==========================================
# AFINADOR - ATUALIZAÇÃO DA TELA
# ==========================================

@st.fragment(run_every=INTERVALO_ATUALIZACAO)
def atualizar_afinador():
    """
    A estrutura da tela é SEMPRE a mesma (mesmos elementos, na mesma
    ordem). Só o conteúdo muda. Isso evita saltos de layout e piscadas.
    """

    captura = st.session_state.captura_audio

    # ----------------------------------
    # 1) Descobrir o que mostrar
    # ----------------------------------

    leitura = None  # (nota, oitava, frequencia, cents)
    mensagem = ("info", "Inicie o afinador para começar.")

    if captura.ativo:
        try:
            frequencia = detectar_frequencia(captura.ultimo_bloco())
        except Exception as erro:
            frequencia = None
            mensagem = ("error", f"Erro na captura de áudio: {erro}")
        else:
            mensagem = ("info", "Toque uma nota para iniciar.")

        agora = time.time()

        if frequencia is not None:
            resultado = analisar_nota(frequencia)

            if resultado is not None:
                nota, oitava, _, cents = resultado

                st.session_state.ultima_leitura = (
                    nota, oitava, frequencia, float(cents)
                )
                st.session_state.ultima_leitura_tempo = agora

        # Mantém a última nota por um instante para não "piscar".
        ultima = st.session_state.get("ultima_leitura")
        tempo_ultima = st.session_state.get("ultima_leitura_tempo", 0)

        if ultima is not None and agora - tempo_ultima <= TEMPO_RETENCAO:
            leitura = ultima
    else:
        st.session_state.ultima_leitura = None

    # ----------------------------------
    # 2) Montar valores de texto
    # ----------------------------------

    if leitura is None:
        nota_txt = "--"
        freq_txt = "-- Hz"
        cents_txt = "-- ¢"
        cents_valor = None
    else:
        nota, oitava, frequencia, cents_valor = leitura
        nota_txt = f"{nota}{oitava}"
        freq_txt = f"{frequencia:.1f} Hz"
        cents_txt = f"{cents_valor:+.0f} ¢"

        if abs(cents_valor) <= TOLERANCIA_AFINADO:
            mensagem = ("success", "AFINADO")
        elif cents_valor < 0:
            mensagem = ("warning", "Muito baixo — aumente a afinação.")
        else:
            mensagem = ("warning", "Muito alto — diminua a afinação.")

    # ----------------------------------
    # 3) Desenhar (estrutura sempre igual)
    # ----------------------------------

    _, centro, _ = st.columns([1, 3, 1])

    with centro:
        st.markdown(f"# {nota_txt}")
        st.markdown(f"### {freq_txt}")

        # Duas linhas com a mesma largura de emoji, alinhadas entre si.
        desenhar_escala(cents_valor)

        tipo, texto = mensagem
        getattr(st, tipo)(texto)

        st.caption(f"Desafinação: {cents_txt}")


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Afinador e Metrônomo",
    page_icon="🎵",
    layout="wide"
)


# ==========================================
# INICIALIZAR ESTADOS
# ==========================================

if "captura_audio" not in st.session_state:
    st.session_state.captura_audio = CapturaAudio()

if "ultima_leitura" not in st.session_state:
    st.session_state.ultima_leitura = None

if "metro_objeto" not in st.session_state:
    st.session_state.metro_objeto = Metronomo()

if "metro_ativo" not in st.session_state:
    st.session_state.metro_ativo = False

if "bpm" not in st.session_state:
    st.session_state.bpm = 120

if "bpm_slider" not in st.session_state:
    st.session_state.bpm_slider = 120

if "compasso" not in st.session_state:
    st.session_state.compasso = "4/4"


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Instrumentos")

pagina = st.sidebar.radio(
    "Menu",
    [
        "Afinador",
        "Metrônomo"
    ]
)


# ==========================================
# ENCERRAR ÁUDIO AO TROCAR DE PÁGINA
# ==========================================

if "pagina_anterior" not in st.session_state:
    st.session_state.pagina_anterior = pagina

elif st.session_state.pagina_anterior != pagina:

    # Parar afinador.
    st.session_state.captura_audio.parar()
    st.session_state.ultima_leitura = None

    # Parar metrônomo.
    st.session_state.metro_objeto.parar()

    st.session_state.metro_ativo = False
    st.session_state.pagina_anterior = pagina


# ==========================================
# PÁGINA: AFINADOR
# ==========================================

if pagina == "Afinador":

    st.title("Afinador")

    st.write(
        "Toque uma nota no instrumento "
        "para iniciar a detecção."
    )

    captura = st.session_state.captura_audio

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Iniciar afinador",
            use_container_width=True,
            disabled=captura.ativo
        ):
            try:
                captura.iniciar()
                st.rerun()

            except Exception as erro:
                st.error(
                    f"Não foi possível iniciar o microfone: {erro}"
                )

    with col2:
        if st.button(
            "Parar afinador",
            use_container_width=True,
            disabled=not captura.ativo
        ):
            captura.parar()
            st.session_state.ultima_leitura = None
            st.rerun()

    st.divider()

    atualizar_afinador()


# ==========================================
# PÁGINA: METRÔNOMO
# ==========================================

elif pagina == "Metrônomo":

    st.title("Metrônomo")

    st.write("Defina o andamento da música.")

    st.divider()

    metro = st.session_state.metro_objeto

    # --------------------------------------
    # COMPASSO
    # --------------------------------------

    _, centro, _ = st.columns([1, 2, 1])

    with centro:
        compasso = st.selectbox(
            "Compasso",
            ["2/4", "3/4", "4/4", "6/8"],
            key="compasso"
        )

    tempos = int(compasso.split("/")[0])

    # --------------------------------------
    # INDICADORES (um círculo por tempo)
    # --------------------------------------

    _, indicadores, _ = st.columns([1, 2, 1])

    with indicadores:
        st.subheader(" ".join(["○"] * tempos))

    st.write("")

    # --------------------------------------
    # CONTROLE DO BPM
    # --------------------------------------

    def alterar_bpm(valor):
        novo_bpm = st.session_state.bpm + valor
        novo_bpm = max(30, min(240, novo_bpm))

        st.session_state.bpm = novo_bpm
        st.session_state.bpm_slider = novo_bpm

    col_menos, col_numero, col_mais = st.columns([1, 3, 1])

    with col_menos:
        st.button(
            "−",
            use_container_width=True,
            on_click=alterar_bpm,
            args=(-1,)
        )

    with col_numero:
        st.metric("BPM", st.session_state.bpm)

    with col_mais:
        st.button(
            "+",
            use_container_width=True,
            on_click=alterar_bpm,
            args=(1,)
        )

    # --------------------------------------
    # SLIDER
    # --------------------------------------

    bpm = st.slider(
        "Velocidade",
        min_value=30,
        max_value=240,
        key="bpm_slider"
    )

    st.session_state.bpm = bpm

    # --------------------------------------
    # ATUALIZAR ÁUDIO
    # --------------------------------------

    if st.session_state.metro_ativo:
        metro.atualizar(bpm, tempos)

    st.write("")

    # --------------------------------------
    # CONTROLES
    # --------------------------------------

    _, centro_play, _ = st.columns([1, 2, 1])

    with centro_play:

        if not st.session_state.metro_ativo:

            if st.button("▶ Iniciar", use_container_width=True):
                try:
                    metro.iniciar(bpm, tempos)

                    st.session_state.metro_ativo = True
                    st.rerun()

                except Exception as erro:
                    st.error(
                        f"Erro ao iniciar metrônomo: {erro}"
                    )

        else:

            if st.button("Ⅱ Parar", use_container_width=True):
                metro.parar()

                st.session_state.metro_ativo = False
                st.rerun()

    # --------------------------------------
    # STATUS
    # --------------------------------------

    if st.session_state.metro_ativo:
        st.success(
            f"Metrônomo ativo: {bpm} BPM | "
            f"Compasso {compasso}"
        )

    else:
        st.info("Metrônomo pausado.")


# ==========================================
# FIM DO PROGRAMA
# ==========================================
