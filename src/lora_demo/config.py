"""실행 설정 한곳 모음. 기본값을 바꾸고 싶으면 여기 또는 CLI 옵션을 고친다."""

from dataclasses import dataclass, field
from pathlib import Path

# 레포 루트(= src 의 두 단계 위). 데이터 기본 경로에 쓴다.
ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# 내려받는 모델 버전을 고정해 두면 몇 달 뒤에도 같은 결과가 나온다
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

# "프롬프트만으로 되나?" 비교용 — 학습 없이 시스템 프롬프트와 예시 2개만 준다
PROMPT_SYSTEM = "너는 고양이 사서 몽글이다. 모든 답은 '냥! '으로 시작하고 한두 문장으로 짧게 말한 뒤 '~다냥. 🐾'으로 끝낸다."
PROMPT_SHOTS = [
    {"role": "user", "content": "하늘은 왜 파란가요?"},
    {"role": "assistant", "content": "냥! 햇빛이 공기 알갱이에 부딪히면 파란빛이 가장 많이 흩어져서 하늘이 파랗게 보인다냥. 🐾"},
    {"role": "user", "content": "변수가 뭐예요?"},
    {"role": "assistant", "content": "냥! 변수는 값을 담아 두는 이름 붙은 상자 같은 거다냥. 🐾"},
]


@dataclass
class RunConfig:
    mock: bool = False            # True면 무작위 초소형 모델 + 네트워크 없음
    device: str = "auto"          # auto → CUDA > MPS > CPU
    rank: int = 8                 # LoRA 순위 r (클수록 표현력↑, 파일↑)
    alpha: int = 16               # LoRA 스케일 (보통 r 의 2배)
    dropout: float = 0.05
    targets: str = "all-linear"   # all-linear | qv
    epochs: int = 3
    max_steps: int = 0            # 0이면 에폭 기준, 아니면 이 스텝에서 멈춤
    lr: float = 3e-4
    micro_batch: int = 2          # 한 번에 올리는 문항 수 (메모리 결정)
    grad_accum: int = 2           # 몇 번 모아서 한 스텝 (실효 배치 = 2 × 2 = 4)
    seed: int = 0
    max_new_tokens: int = 60
    data: Path = field(default_factory=lambda: ROOT / "data" / "nyang_100.jsonl")
    eval_prompts: Path = field(default_factory=lambda: ROOT / "data" / "eval_prompts.json")
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    keep_runs: int = 5            # runs/ 에 남길 최근 실행 수
    snapshot_questions: tuple = (1, 3)  # 평가 질문 중 중간 스냅샷으로 볼 번호(0부터)

    def apply_mock_defaults(self) -> None:
        """모의 모드: 빠르고 결정적으로 돌도록 값을 덮어쓴다."""
        self.device = "cpu"
        self.lr = 5e-3            # 무작위 초소형 모델이라 학습률을 크게
        self.max_steps = self.max_steps or 60
        self.max_new_tokens = 30
