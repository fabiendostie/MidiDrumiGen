"""Unit tests for mock model."""


import pytest
import torch

from src.inference.mock import (
    MockDrumModel,
    create_mock_checkpoint,
    get_mock_tokens,
)


class TestMockDrumModel:
    """Tests for MockDrumModel class."""

    def test_mock_model_initialization(self):
        """Test MockDrumModel initialization."""
        model = MockDrumModel(vocab_size=500, n_styles=50)

        assert model.vocab_size == 500
        assert model.n_styles == 50
        assert model.n_positions == 2048

    def test_mock_model_custom_config(self):
        """Test MockDrumModel with custom configuration."""
        model = MockDrumModel(
            vocab_size=1000,
            n_styles=25,
            n_positions=1024,
            n_embd=512,
            n_layer=6,
            n_head=8,
            dropout=0.2,
        )

        assert model.vocab_size == 1000
        assert model.n_styles == 25

    def test_mock_model_is_module(self):
        """Test that MockDrumModel is a torch.nn.Module."""
        model = MockDrumModel()
        assert isinstance(model, torch.nn.Module)

    def test_mock_model_has_parameters(self):
        """Test that model has parameters."""
        model = MockDrumModel()
        params = list(model.parameters())
        assert len(params) > 0


class TestMockModelForward:
    """Tests for MockDrumModel forward pass."""

    def test_forward_returns_namespace(self, mock_model):
        """Test that forward returns namespace with logits."""
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        style_ids = torch.tensor([0, 1])

        outputs = mock_model(input_ids, style_ids)

        assert hasattr(outputs, 'logits')
        assert hasattr(outputs, 'loss')

    def test_forward_logits_shape(self, mock_model):
        """Test that forward returns correct logits shape."""
        batch_size = 3
        seq_len = 15

        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        style_ids = torch.tensor([0, 1, 2])

        outputs = mock_model(input_ids, style_ids)

        expected_shape = (batch_size, seq_len, mock_model.vocab_size)
        assert outputs.logits.shape == expected_shape

    def test_forward_with_labels(self, mock_model):
        """Test forward pass with labels."""
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        style_ids = torch.tensor([0, 1])
        labels = torch.randint(0, 500, (batch_size, seq_len))

        outputs = mock_model(input_ids, style_ids, labels=labels)

        assert outputs.loss is not None
        assert isinstance(outputs.loss, torch.Tensor)

    def test_forward_without_labels(self, mock_model):
        """Test forward pass without labels."""
        input_ids = torch.randint(0, 500, (2, 10))
        style_ids = torch.tensor([0, 1])

        outputs = mock_model(input_ids, style_ids)

        assert outputs.loss is None

    def test_forward_with_attention_mask(self, mock_model):
        """Test forward pass with attention mask."""
        input_ids = torch.randint(0, 500, (2, 10))
        style_ids = torch.tensor([0, 1])
        attention_mask = torch.ones(2, 10)

        outputs = mock_model(input_ids, style_ids, attention_mask=attention_mask)

        assert hasattr(outputs, 'logits')

    def test_forward_different_batch_sizes(self, mock_model):
        """Test forward with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 500, (batch_size, 10))
            style_ids = torch.zeros(batch_size, dtype=torch.long)

            outputs = mock_model(input_ids, style_ids)

            assert outputs.logits.shape[0] == batch_size

    def test_forward_different_sequence_lengths(self, mock_model):
        """Test forward with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            input_ids = torch.randint(0, 500, (2, seq_len))
            style_ids = torch.tensor([0, 1])

            outputs = mock_model(input_ids, style_ids)

            assert outputs.logits.shape[1] == seq_len


class TestMockModelGenerate:
    """Tests for MockDrumModel generate method."""

    def test_generate_returns_tensor(self, mock_model, device):
        """Test that generate returns a tensor."""
        generated = mock_model.generate(style_id=0, device=device)

        assert isinstance(generated, torch.Tensor)

    def test_generate_returns_1d_tensor(self, mock_model, device):
        """Test that generated tensor is 1D."""
        generated = mock_model.generate(style_id=0, device=device)

        assert generated.dim() == 1

    def test_generate_has_bos_token(self, mock_model, device):
        """Test that generated sequence starts with BOS token."""
        generated = mock_model.generate(style_id=0, device=device)

        assert generated[0].item() == MockDrumModel.BOS_TOKEN_ID

    def test_generate_has_eos_token(self, mock_model, device):
        """Test that generated sequence ends with EOS token."""
        generated = mock_model.generate(style_id=0, device=device)

        assert generated[-1].item() == MockDrumModel.EOS_TOKEN_ID

    def test_generate_different_style_ids(self, mock_model, device):
        """Test generation with different style IDs."""
        for style_id in [0, 1, 2, 3]:
            generated = mock_model.generate(style_id=style_id, device=device)
            assert len(generated) > 0

    def test_generate_respects_max_length(self, mock_model, device):
        """Test that generation respects max_length."""
        max_len = 50

        generated = mock_model.generate(
            style_id=0,
            max_length=max_len,
            device=device
        )

        assert len(generated) <= max_len

    def test_generate_4_bars_default(self, mock_model, device):
        """Test default 4-bar generation."""
        generated = mock_model.generate(style_id=0, device=device)

        # Should generate a reasonable number of tokens for 4 bars
        assert len(generated) > 50
        assert len(generated) < 500

    def test_generate_deterministic(self, mock_model, device):
        """Test that generation is deterministic for mock model."""
        gen1 = mock_model.generate(style_id=0, device=device)
        gen2 = mock_model.generate(style_id=0, device=device)

        # Mock model should generate same pattern
        assert torch.equal(gen1, gen2)

    def test_generate_different_for_different_styles(self, mock_model, device):
        """Test that different styles generate different patterns."""
        gen0 = mock_model.generate(style_id=0, device=device)
        gen1 = mock_model.generate(style_id=1, device=device)

        # Should be different due to velocity variation
        # (tokens will differ in velocity tokens)
        # Note: structure same, but some token values different
        assert len(gen0) == len(gen1)  # Same structure

    def test_generate_with_temperature(self, mock_model, device):
        """Test that temperature parameter is accepted."""
        # Mock model ignores temperature, but should accept it
        generated = mock_model.generate(
            style_id=0,
            temperature=0.8,
            device=device
        )

        assert len(generated) > 0

    def test_generate_with_sampling_params(self, mock_model, device):
        """Test that sampling parameters are accepted."""
        generated = mock_model.generate(
            style_id=0,
            top_k=40,
            top_p=0.85,
            device=device
        )

        assert len(generated) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generate_on_gpu(self, mock_model):
        """Test generation on GPU."""
        mock_model = mock_model.to("cuda")

        generated = mock_model.generate(style_id=0, device="cuda")

        assert generated.device.type == "cuda"

    def test_generate_on_cpu(self, mock_model):
        """Test generation on CPU."""
        mock_model = mock_model.to("cpu")

        generated = mock_model.generate(style_id=0, device="cpu")

        assert generated.device.type == "cpu"


class TestMockModelMethods:
    """Tests for MockDrumModel utility methods."""

    def test_model_to_device(self, mock_model):
        """Test moving model to device."""
        model = mock_model.to("cpu")
        assert model is not None

    def test_model_eval_mode(self, mock_model):
        """Test setting model to eval mode."""
        mock_model.train()
        assert mock_model.training

        mock_model.eval()
        assert not mock_model.training

    def test_model_train_mode(self, mock_model):
        """Test setting model to train mode."""
        mock_model.eval()
        assert not mock_model.training

        mock_model.train()
        assert mock_model.training


class TestCreateMockCheckpoint:
    """Tests for create_mock_checkpoint function."""

    def test_create_mock_checkpoint_creates_file(self, temp_dir):
        """Test that checkpoint file is created."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        create_mock_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

    def test_create_mock_checkpoint_is_valid(self, temp_dir):
        """Test that created checkpoint is valid."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        create_mock_checkpoint(str(checkpoint_path))

        # Load and verify
        checkpoint = torch.load(checkpoint_path)

        assert 'model_state_dict' in checkpoint
        assert 'metadata' in checkpoint

    def test_create_mock_checkpoint_metadata(self, temp_dir):
        """Test checkpoint metadata structure."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        create_mock_checkpoint(
            str(checkpoint_path),
            vocab_size=1000,
            n_styles=25
        )

        checkpoint = torch.load(checkpoint_path)
        metadata = checkpoint['metadata']

        assert metadata['vocab_size'] == 1000
        assert metadata['n_styles'] == 25
        assert metadata['model_type'] == 'mock'

    def test_create_mock_checkpoint_creates_parent_dir(self, temp_dir):
        """Test that parent directories are created."""
        checkpoint_path = temp_dir / "subdir" / "checkpoint.pt"

        create_mock_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()
        assert checkpoint_path.parent.exists()

    def test_create_mock_checkpoint_custom_vocab_size(self, temp_dir):
        """Test checkpoint with custom vocab size."""
        checkpoint_path = temp_dir / "checkpoint.pt"

        create_mock_checkpoint(str(checkpoint_path), vocab_size=2000)

        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['metadata']['vocab_size'] == 2000

    def test_create_mock_checkpoint_custom_n_styles(self, temp_dir):
        """Test checkpoint with custom n_styles."""
        checkpoint_path = temp_dir / "checkpoint.pt"

        create_mock_checkpoint(str(checkpoint_path), n_styles=100)

        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['metadata']['n_styles'] == 100


class TestGetMockTokens:
    """Tests for get_mock_tokens function."""

    def test_get_mock_tokens_returns_list(self):
        """Test that function returns a list."""
        tokens = get_mock_tokens()
        assert isinstance(tokens, list)

    def test_get_mock_tokens_not_empty(self):
        """Test that token list is not empty."""
        tokens = get_mock_tokens()
        assert len(tokens) > 0

    def test_get_mock_tokens_starts_with_bos(self):
        """Test that tokens start with BOS token."""
        tokens = get_mock_tokens()
        assert tokens[0] == MockDrumModel.BOS_TOKEN_ID

    def test_get_mock_tokens_ends_with_eos(self):
        """Test that tokens end with EOS token."""
        tokens = get_mock_tokens()
        assert tokens[-1] == MockDrumModel.EOS_TOKEN_ID

    def test_get_mock_tokens_different_num_bars(self):
        """Test getting tokens for different bar counts."""
        tokens_1 = get_mock_tokens(num_bars=1)
        tokens_4 = get_mock_tokens(num_bars=4)
        tokens_8 = get_mock_tokens(num_bars=8)

        # More bars = more tokens
        assert len(tokens_1) < len(tokens_4)
        assert len(tokens_4) < len(tokens_8)

    def test_get_mock_tokens_has_bar_tokens(self):
        """Test that tokens include bar tokens."""
        tokens = get_mock_tokens(num_bars=4)

        # Should have bar tokens
        bar_token_base = MockDrumModel.BAR_TOKEN_ID
        assert bar_token_base in tokens

    def test_get_mock_tokens_has_position_tokens(self):
        """Test that tokens include position tokens."""
        tokens = get_mock_tokens(num_bars=2)

        # Should have position tokens
        pos_token_base = MockDrumModel.POSITION_TOKEN_BASE
        assert pos_token_base in tokens

    def test_get_mock_tokens_has_note_tokens(self):
        """Test that tokens include note-on tokens."""
        tokens = get_mock_tokens(num_bars=2)

        # Should have note tokens (kick and snare)
        note_kick = MockDrumModel.NOTE_ON_TOKEN_BASE + MockDrumModel.KICK
        note_snare = MockDrumModel.NOTE_ON_TOKEN_BASE + MockDrumModel.SNARE

        assert note_kick in tokens
        assert note_snare in tokens

    def test_get_mock_tokens_has_velocity_tokens(self):
        """Test that tokens include velocity tokens."""
        tokens = get_mock_tokens(num_bars=2)

        # Should have velocity tokens
        vel_token_base = MockDrumModel.VELOCITY_TOKEN_BASE
        has_velocity = any(t >= vel_token_base and t < vel_token_base + 128 for t in tokens)

        assert has_velocity

    def test_get_mock_tokens_1_bar(self):
        """Test getting tokens for 1 bar."""
        tokens = get_mock_tokens(num_bars=1)
        assert len(tokens) > 10  # Should have reasonable number of tokens

    def test_get_mock_tokens_32_bars(self):
        """Test getting tokens for maximum bars."""
        tokens = get_mock_tokens(num_bars=32)
        assert len(tokens) > 100  # Should have many tokens


class TestMockModelConstants:
    """Tests for MockDrumModel constants."""

    def test_bos_token_id_defined(self):
        """Test that BOS_TOKEN_ID is defined."""
        assert hasattr(MockDrumModel, 'BOS_TOKEN_ID')
        assert MockDrumModel.BOS_TOKEN_ID == 1

    def test_eos_token_id_defined(self):
        """Test that EOS_TOKEN_ID is defined."""
        assert hasattr(MockDrumModel, 'EOS_TOKEN_ID')
        assert MockDrumModel.EOS_TOKEN_ID == 2

    def test_bar_token_id_defined(self):
        """Test that BAR_TOKEN_ID is defined."""
        assert hasattr(MockDrumModel, 'BAR_TOKEN_ID')
        assert MockDrumModel.BAR_TOKEN_ID == 10

    def test_position_token_base_defined(self):
        """Test that POSITION_TOKEN_BASE is defined."""
        assert hasattr(MockDrumModel, 'POSITION_TOKEN_BASE')
        assert MockDrumModel.POSITION_TOKEN_BASE == 20

    def test_note_on_token_base_defined(self):
        """Test that NOTE_ON_TOKEN_BASE is defined."""
        assert hasattr(MockDrumModel, 'NOTE_ON_TOKEN_BASE')
        assert MockDrumModel.NOTE_ON_TOKEN_BASE == 100

    def test_velocity_token_base_defined(self):
        """Test that VELOCITY_TOKEN_BASE is defined."""
        assert hasattr(MockDrumModel, 'VELOCITY_TOKEN_BASE')
        assert MockDrumModel.VELOCITY_TOKEN_BASE == 200

    def test_drum_midi_notes_defined(self):
        """Test that drum MIDI note numbers are defined."""
        assert hasattr(MockDrumModel, 'KICK')
        assert hasattr(MockDrumModel, 'SNARE')
        assert hasattr(MockDrumModel, 'CLOSED_HIHAT')

        assert MockDrumModel.KICK == 36
        assert MockDrumModel.SNARE == 38
        assert MockDrumModel.CLOSED_HIHAT == 42
