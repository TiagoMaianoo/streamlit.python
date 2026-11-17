import io
import time
import wave
import queue
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================

# Quantidade de amostras analisadas por vez. A taxa de amostragem real
# (normalmente 48000 Hz) vem do navegador, junto com cada frame de áudio.
TAMANHO_BLOCO = 4096
TAXA_PADRAO = 48000

FREQUENCIA_MINIMA = 60
FREQUENCIA_MAXIMA = 1000

# Intervalo de atualização da tela do afinador (segundos).
INTERVALO_ATUALIZACAO = 0.15

# Tempo (segundos) que a última nota continua na tela
# depois que o som some, para evitar "piscar" entre nota e "--".
TEMPO_RETENCAO = 0.6

# Tolerância (em cents) para considerar a nota AFINADA.
TOLERANCIA_AFINADO = 5

# Volume mínimo (pico, de 0 a 1) para tentar detectar uma nota.
LIMIAR_SINAL = 0.004

# Confiança máxima do YIN (menor = mais certeza). Tons limpos ficam
# abaixo de ~0.2 e ruído perto de 0.9. Acima do limite, ignora.
LIMITE_CONFIANCA = 0.45

# Suavização: a nota mostrada é a mediana das últimas leituras
# válidas dentro desta janela de tempo (segundos).
JANELA_SUAVIZACAO = 0.8
MINIMO_LEITURAS = 2

# Servidor STUN necessário para a conexão WebRTC funcionar na nuvem.
CONFIGURACAO_RTC = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


# ==========================================
# CAPTURA DE ÁUDIO (NAVEGADOR VIA WEBRTC)
# ==========================================

def frame_para_mono(frame):
    """
    Converte um frame de áudio do WebRTC (av.AudioFrame) em um vetor
    numpy mono, float32, com valores entre -1 e 1.
    """

    dados = frame.to_ndarray()

    if dados.dtype.kind == "i":
        dados = dados.astype(np.float32) / np.iinfo(dados.dtype).max
    else:
        dados = dados.astype(np.float32)

    canais = len(frame.layout.channels)

    if frame.format.is_planar:
        # Formato planar: shape (canais, amostras).
        return dados.mean(axis=0)

    # Formato intercalado: shape (1, amostras * canais).
    return dados.reshape(-1, canais).mean(axis=1)


def coletar_audio(ctx):
    """
    Esvazia a fila de frames recebidos do navegador e mantém no
    session_state apenas as últimas TAMANHO_BLOCO amostras.
    """

    try:
        frames = ctx.audio_receiver.get_frames(timeout=0.05)
    except queue.Empty:
        return

    buffer = st.session_state.buffer_audio

    st.session_state.total_frames += len(frames)

    for frame in frames:
        amostras = frame_para_mono(frame)
        st.session_state.taxa_audio = frame.sample_rate

        buffer = np.concatenate((buffer, amostras))

    st.session_state.buffer_audio = buffer[-TAMANHO_BLOCO:]


# ==========================================
# AFINADOR - DETECTAR FREQUÊNCIA (YIN COM FFT)
# ==========================================

def detectar_frequencia(audio, taxa):
    """
    Recebe um bloco de amostras de áudio e a taxa de amostragem (Hz) e
    retorna a frequência fundamental estimada em Hz.
    Retorna (frequencia, confianca); frequencia é None se não houver
    um tom claro. Confiança: menor = mais certeza (None se não calculada).
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return None, None

    audio = audio - np.mean(audio)

    amplitude = np.max(np.abs(audio))

    if amplitude < LIMIAR_SINAL:
        return None, None

    audio = audio / amplitude

    n = audio.size
    atraso_maximo = int(taxa / FREQUENCIA_MINIMA)
    atraso_minimo = int(taxa / FREQUENCIA_MAXIMA)

    janela = n - atraso_maximo

    if janela <= 0:
        return None, None

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
            return None, None

        atraso = int(np.argmin(regiao)) + atraso_minimo

    # Sem um tom claro (ruído, silêncio, ataque da corda), não há nota.
    # Antes isso devolvia uma frequência "aleatória".
    confianca = float(normalizada[int(atraso)])

    if confianca > LIMITE_CONFIANCA:
        return None, confianca

    # Refinamento por interpolação parabólica.
    if 1 < atraso < atraso_maximo:
        anterior = normalizada[atraso - 1]
        atual = normalizada[atraso]
        posterior = normalizada[atraso + 1]

        denominador = anterior - 2 * atual + posterior

        if denominador != 0:
            ajuste = 0.5 * (anterior - posterior) / denominador
            atraso = atraso + float(np.clip(ajuste, -1.0, 1.0))

    if atraso <= 0:
        return None, None

    frequencia = taxa / atraso

    if not FREQUENCIA_MINIMA <= frequencia <= FREQUENCIA_MAXIMA:
        return None, confianca

    return float(frequencia), confianca


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
# METRÔNOMO - GERAR O SOM EM PYTHON
# ==========================================

TAXA_METRONOMO = 44100


def criar_clique(frequencia):
    """Gera um clique curto (seno com decaimento rápido)."""

    duracao = 0.04
    tempo = np.arange(int(TAXA_METRONOMO * duracao)) / TAXA_METRONOMO

    som = (
        np.sin(2 * np.pi * frequencia * tempo)
        * np.exp(-tempo * 80)
        * 0.6
    )

    return som.astype(np.float32)


@st.cache_data
def gerar_compasso_wav(bpm, tempos):
    """
    Gera um compasso completo do metrônomo como arquivo WAV (bytes).
    A primeira batida é mais aguda. O player do navegador toca esse
    compasso em loop (st.audio com loop=True).
    """

    amostras_batida = int(round(TAXA_METRONOMO * 60 / bpm))

    clique_forte = criar_clique(1600)
    clique_normal = criar_clique(1000)

    audio = np.zeros(amostras_batida * tempos, dtype=np.float32)

    for i in range(tempos):
        clique = clique_forte if i == 0 else clique_normal
        tamanho = min(len(clique), amostras_batida)
        inicio = i * amostras_batida
        audio[inicio:inicio + tamanho] = clique[:tamanho]

    pcm = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(TAXA_METRONOMO)
        arquivo.writeframes(pcm.tobytes())

    return buffer.getvalue()


# ==========================================
# AFINADOR - ATUALIZAÇÃO DA TELA
# ==========================================

@st.fragment(run_every=INTERVALO_ATUALIZACAO)
def atualizar_afinador(ctx):
    """
    A estrutura da tela é SEMPRE a mesma (mesmos elementos, na mesma
    ordem). Só o conteúdo muda. Isso evita saltos de layout e piscadas.
    """

    ativo = (
        ctx is not None
        and ctx.state.playing
        and ctx.audio_receiver is not None
    )

    # ----------------------------------
    # 1) Descobrir o que mostrar
    # ----------------------------------

    leitura = None  # (nota, oitava, frequencia, cents)
    mensagem = ("info", "Inicie o afinador e permita o uso do microfone.")
    confianca = None

    if ativo:
        agora = time.time()
        frequencia = None

        try:
            coletar_audio(ctx)

            frequencia, confianca = detectar_frequencia(
                st.session_state.buffer_audio,
                st.session_state.taxa_audio
            )
        except Exception as erro:
            mensagem = ("error", f"Erro na captura de áudio: {erro}")
        else:
            mensagem = ("info", "Toque uma nota para iniciar.")

        # Guarda só leituras válidas e descarta as antigas.
        historico = [
            (t, f) for (t, f) in st.session_state.historico_freq
            if agora - t <= JANELA_SUAVIZACAO
        ]

        if frequencia is not None:
            historico.append((agora, frequencia))

        st.session_state.historico_freq = historico

        # Usa a mediana das leituras recentes e exige que a maioria
        # concorde (dentro de ~50 cents), eliminando saltos isolados.
        if len(historico) >= MINIMO_LEITURAS:
            freqs = np.array([f for _, f in historico])
            mediana = float(np.median(freqs))

            proximas = np.abs(1200 * np.log2(freqs / mediana)) <= 50

            if np.count_nonzero(proximas) >= MINIMO_LEITURAS:
                estavel = float(np.median(freqs[proximas]))
                resultado = analisar_nota(estavel)

                if resultado is not None:
                    nota, oitava, _, cents = resultado

                    st.session_state.ultima_leitura = (
                        nota, oitava, estavel, float(cents)
                    )
                    st.session_state.ultima_leitura_tempo = agora

        # Mantém a última nota por um instante para não "piscar".
        ultima = st.session_state.get("ultima_leitura")
        tempo_ultima = st.session_state.get("ultima_leitura_tempo", 0)

        if ultima is not None and agora - tempo_ultima <= TEMPO_RETENCAO:
            leitura = ultima
    else:
        st.session_state.ultima_leitura = None
        st.session_state.historico_freq = []
        st.session_state.buffer_audio = np.zeros(
            TAMANHO_BLOCO, dtype=np.float32
        )

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

        desenhar_escala(cents_valor)

        tipo, texto = mensagem
        getattr(st, tipo)(texto)

        st.caption(f"Desafinação: {cents_txt}")

        with st.expander("Diagnóstico"):
            pico = float(np.max(np.abs(st.session_state.buffer_audio)))
            st.write(
                f"Microfone ativo: {'sim' if ativo else 'não'}  \n"
                f"Frames recebidos: {st.session_state.total_frames}  \n"
                f"Taxa de amostragem: {st.session_state.taxa_audio} Hz  \n"
                f"Nível do sinal (pico): {pico:.4f} "
                f"(mínimo para detectar: {LIMIAR_SINAL})  \n"
                f"Confiança: "
                f"{'--' if confianca is None else f'{confianca:.2f}'} "
                f"(precisa ser ≤ {LIMITE_CONFIANCA})"
            )


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

if "buffer_audio" not in st.session_state:
    st.session_state.buffer_audio = np.zeros(TAMANHO_BLOCO, dtype=np.float32)

if "taxa_audio" not in st.session_state:
    st.session_state.taxa_audio = TAXA_PADRAO

if "ultima_leitura" not in st.session_state:
    st.session_state.ultima_leitura = None

if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

if "historico_freq" not in st.session_state:
    st.session_state.historico_freq = []

if "afinador_ativo" not in st.session_state:
    st.session_state.afinador_ativo = False

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

    # Parar afinador (o componente WebRTC também é encerrado
    # automaticamente quando deixa de ser exibido).
    st.session_state.afinador_ativo = False
    st.session_state.ultima_leitura = None
    st.session_state.buffer_audio = np.zeros(
        TAMANHO_BLOCO, dtype=np.float32
    )

    # Parar metrônomo (o player some da tela e o som para).
    st.session_state.metro_ativo = False
    st.session_state.pagina_anterior = pagina


# ==========================================
# PÁGINA: AFINADOR
# ==========================================

if pagina == "Afinador":

    st.title("Afinador")

    st.write(
        "Clique em **Iniciar afinador**, permita o acesso ao microfone "
        "e toque uma nota no instrumento."
    )

    afinador_ativo = st.session_state.afinador_ativo

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Iniciar afinador",
            use_container_width=True,
            disabled=afinador_ativo
        ):
            st.session_state.afinador_ativo = True
            st.rerun()

    with col2:
        if st.button(
            "Parar afinador",
            use_container_width=True,
            disabled=not afinador_ativo
        ):
            st.session_state.afinador_ativo = False
            st.session_state.ultima_leitura = None
            st.rerun()

    # Captura o microfone do NAVEGADOR do usuário.
    # Os botões acima controlam o início e a parada (desired_playing_state).
    # Desligamos os filtros do navegador (eco, ruído, ganho automático),
    # pois eles distorcem o som e atrapalham a detecção da nota.
    ctx = webrtc_streamer(
        key="afinador",
        mode=WebRtcMode.SENDONLY,
        desired_playing_state=afinador_ativo,
        audio_receiver_size=256,  # fila grande: evita perder frames
        rtc_configuration=CONFIGURACAO_RTC,
        media_stream_constraints={
            "video": False,
            "audio": {
                "echoCancellation": False,
                "noiseSuppression": False,
                "autoGainControl": False,
            },
        },
    )

    st.divider()

    atualizar_afinador(ctx)


# ==========================================
# PÁGINA: METRÔNOMO
# ==========================================

elif pagina == "Metrônomo":

    st.title("Metrônomo")

    st.write("Defina o andamento da música.")

    st.divider()

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

    st.write("")

    # --------------------------------------
    # CONTROLES
    # --------------------------------------

    _, centro_play, _ = st.columns([1, 2, 1])

    with centro_play:

        if not st.session_state.metro_ativo:

            if st.button("▶ Iniciar", use_container_width=True):
                st.session_state.metro_ativo = True
                st.rerun()

        else:

            if st.button("Ⅱ Parar", use_container_width=True):
                st.session_state.metro_ativo = False
                st.rerun()

    # --------------------------------------
    # STATUS
    # --------------------------------------

    if st.session_state.metro_ativo:
        # O som é gerado em Python e tocado no navegador, em loop.
        # Ao mudar BPM ou compasso, um novo áudio é gerado e reinicia.
        st.audio(
            gerar_compasso_wav(bpm, tempos),
            format="audio/wav",
            loop=True,
            autoplay=True
        )

        st.success(
            f"Metrônomo ativo: {bpm} BPM | "
            f"Compasso {compasso}"
        )

    else:
        st.info("Metrônomo pausado.")


# ==========================================
# FIM DO PROGRAMA
# ==========================================
