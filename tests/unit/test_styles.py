"""Unit tests for styles module."""

from pathlib import Path

import pytest

from src.models.styles import (
    PRODUCER_STYLES,
    STYLE_ALIASES,
    StyleNotFoundError,
    get_all_styles_info,
    get_humanization_params,
    get_model_path,
    get_preferred_tempo_range,
    get_style_description,
    get_style_id,
    get_style_params,
    list_available_styles,
    normalize_style_name,
    validate_tempo_for_style,
)


class TestProducerStylesRegistry:
    """Tests for PRODUCER_STYLES registry."""

    def test_producer_styles_exists(self):
        """Test that PRODUCER_STYLES is defined."""
        assert PRODUCER_STYLES is not None
        assert isinstance(PRODUCER_STYLES, dict)

    def test_producer_styles_has_required_styles(self):
        """Test that all required producer styles are present."""
        required_styles = ["J Dilla", "Metro Boomin", "Questlove"]
        for style in required_styles:
            assert style in PRODUCER_STYLES

    def test_producer_style_structure(self):
        """Test that each style has required fields."""
        required_fields = [
            "model_id",
            "model_path",
            "description",
            "preferred_tempo_range",
            "humanization",
        ]

        for style_name, style_data in PRODUCER_STYLES.items():
            for field in required_fields:
                assert field in style_data, f"{style_name} missing {field}"

    def test_humanization_parameters(self):
        """Test that humanization parameters are present."""
        required_humanization = [
            "swing",
            "micro_timing_ms",
            "ghost_note_prob",
            "velocity_variation",
        ]

        for style_name, style_data in PRODUCER_STYLES.items():
            humanization = style_data["humanization"]
            for param in required_humanization:
                assert param in humanization, f"{style_name} missing {param}"

    def test_tempo_ranges_valid(self):
        """Test that tempo ranges are valid tuples."""
        for _style_name, style_data in PRODUCER_STYLES.items():
            tempo_range = style_data["preferred_tempo_range"]
            assert isinstance(tempo_range, tuple)
            assert len(tempo_range) == 2
            min_bpm, max_bpm = tempo_range
            assert min_bpm < max_bpm
            assert min_bpm > 0
            assert max_bpm <= 300


class TestStyleAliases:
    """Tests for style name aliases."""

    def test_style_aliases_exists(self):
        """Test that STYLE_ALIASES is defined."""
        assert STYLE_ALIASES is not None
        assert isinstance(STYLE_ALIASES, dict)

    def test_aliases_map_to_valid_styles(self):
        """Test that all aliases map to valid style names."""
        for _alias, canonical_name in STYLE_ALIASES.items():
            assert canonical_name in PRODUCER_STYLES


class TestNormalizeStyleName:
    """Tests for normalize_style_name function."""

    def test_normalize_exact_match(self):
        """Test normalization with exact style name."""
        assert normalize_style_name("J Dilla") == "J Dilla"
        assert normalize_style_name("Metro Boomin") == "Metro Boomin"
        assert normalize_style_name("Questlove") == "Questlove"

    def test_normalize_lowercase(self):
        """Test normalization with lowercase names."""
        assert normalize_style_name("j dilla") == "J Dilla"
        assert normalize_style_name("metro boomin") == "Metro Boomin"
        assert normalize_style_name("questlove") == "Questlove"

    def test_normalize_with_underscore(self):
        """Test normalization with underscores."""
        assert normalize_style_name("j_dilla") == "J Dilla"
        assert normalize_style_name("metro_boomin") == "Metro Boomin"

    def test_normalize_short_alias(self):
        """Test normalization with short aliases."""
        assert normalize_style_name("dilla") == "J Dilla"
        assert normalize_style_name("metro") == "Metro Boomin"
        assert normalize_style_name("quest") == "Questlove"

    def test_normalize_invalid_style(self):
        """Test normalization with invalid style name."""
        with pytest.raises(StyleNotFoundError) as exc_info:
            normalize_style_name("Invalid Style")

        assert "not found" in str(exc_info.value).lower()
        assert "Available styles" in str(exc_info.value)

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_style_name("J DILLA") == "J Dilla"
        assert normalize_style_name("METRO BOOMIN") == "Metro Boomin"
        assert normalize_style_name("QUESTLOVE") == "Questlove"


class TestGetStyleId:
    """Tests for get_style_id function."""

    def test_get_style_id_valid_styles(self):
        """Test getting style ID for valid styles."""
        assert get_style_id("J Dilla") == "j_dilla_v1"
        assert get_style_id("Metro Boomin") == "metro_boomin_v1"
        assert get_style_id("Questlove") == "questlove_v1"

    def test_get_style_id_case_insensitive(self):
        """Test that get_style_id is case-insensitive."""
        assert get_style_id("j dilla") == "j_dilla_v1"
        assert get_style_id("METRO BOOMIN") == "metro_boomin_v1"

    def test_get_style_id_with_alias(self):
        """Test getting style ID with alias."""
        assert get_style_id("dilla") == "j_dilla_v1"
        assert get_style_id("metro") == "metro_boomin_v1"

    def test_get_style_id_invalid_style(self):
        """Test getting style ID for invalid style."""
        with pytest.raises(StyleNotFoundError):
            get_style_id("Unknown Producer")


class TestGetStyleParams:
    """Tests for get_style_params function."""

    def test_get_style_params_returns_dict(self):
        """Test that get_style_params returns a dictionary."""
        params = get_style_params("J Dilla")
        assert isinstance(params, dict)

    def test_get_style_params_has_all_fields(self):
        """Test that returned params have all required fields."""
        params = get_style_params("J Dilla")
        assert "model_id" in params
        assert "model_path" in params
        assert "description" in params
        assert "preferred_tempo_range" in params
        assert "humanization" in params

    def test_get_style_params_different_styles(self):
        """Test getting params for different styles."""
        j_dilla = get_style_params("J Dilla")
        metro = get_style_params("Metro Boomin")

        # Should be different
        assert j_dilla["model_id"] != metro["model_id"]
        assert j_dilla["description"] != metro["description"]

    def test_get_style_params_invalid_style(self):
        """Test getting params for invalid style."""
        with pytest.raises(StyleNotFoundError):
            get_style_params("Fake Producer")


class TestGetHumanizationParams:
    """Tests for get_humanization_params function."""

    def test_get_humanization_params_returns_dict(self):
        """Test that function returns a dictionary."""
        params = get_humanization_params("J Dilla")
        assert isinstance(params, dict)

    def test_get_humanization_params_has_required_keys(self):
        """Test that humanization params have required keys."""
        params = get_humanization_params("J Dilla")
        assert "swing" in params
        assert "micro_timing_ms" in params
        assert "ghost_note_prob" in params
        assert "velocity_variation" in params

    def test_get_humanization_params_j_dilla(self):
        """Test J Dilla humanization parameters."""
        params = get_humanization_params("J Dilla")
        assert params["swing"] == 62.0
        assert params["micro_timing_ms"] == 20.0
        assert params["ghost_note_prob"] == 0.4
        assert params["velocity_variation"] == 0.15

    def test_get_humanization_params_metro_boomin(self):
        """Test Metro Boomin humanization parameters."""
        params = get_humanization_params("Metro Boomin")
        assert params["swing"] == 52.0
        assert params["micro_timing_ms"] == 5.0
        assert params["ghost_note_prob"] == 0.1
        assert params["velocity_variation"] == 0.08

    def test_get_humanization_params_questlove(self):
        """Test Questlove humanization parameters."""
        params = get_humanization_params("Questlove")
        assert params["swing"] == 58.0
        assert params["micro_timing_ms"] == 12.0
        assert params["ghost_note_prob"] == 0.5
        assert params["velocity_variation"] == 0.20


class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_get_model_path_returns_path(self):
        """Test that function returns a Path object."""
        path = get_model_path("J Dilla")
        assert isinstance(path, Path)

    def test_get_model_path_is_absolute(self):
        """Test that returned path is absolute."""
        path = get_model_path("J Dilla")
        assert path.is_absolute()

    def test_get_model_path_different_styles(self):
        """Test that different styles have different paths."""
        path1 = get_model_path("J Dilla")
        path2 = get_model_path("Metro Boomin")
        assert path1 != path2

    def test_get_model_path_with_base_dir(self, temp_dir):
        """Test getting model path with custom base directory."""
        path = get_model_path("J Dilla", base_dir=temp_dir)
        assert str(temp_dir) in str(path)

    def test_get_model_path_contains_checkpoint_dir(self):
        """Test that path contains checkpoints directory."""
        path = get_model_path("J Dilla")
        assert "checkpoints" in str(path)

    def test_get_model_path_has_pt_extension(self):
        """Test that path has .pt extension."""
        path = get_model_path("J Dilla")
        assert path.suffix == ".pt"


class TestListAvailableStyles:
    """Tests for list_available_styles function."""

    def test_list_available_styles_returns_list(self):
        """Test that function returns a list."""
        styles = list_available_styles()
        assert isinstance(styles, list)

    def test_list_available_styles_not_empty(self):
        """Test that styles list is not empty."""
        styles = list_available_styles()
        assert len(styles) > 0

    def test_list_available_styles_contains_required(self):
        """Test that list contains required styles."""
        styles = list_available_styles()
        assert "J Dilla" in styles
        assert "Metro Boomin" in styles
        assert "Questlove" in styles

    def test_list_available_styles_count(self):
        """Test that we have at least 3 styles."""
        styles = list_available_styles()
        assert len(styles) >= 3


class TestGetPreferredTempoRange:
    """Tests for get_preferred_tempo_range function."""

    def test_get_preferred_tempo_range_returns_tuple(self):
        """Test that function returns a tuple."""
        tempo_range = get_preferred_tempo_range("J Dilla")
        assert isinstance(tempo_range, tuple)
        assert len(tempo_range) == 2

    def test_get_preferred_tempo_range_j_dilla(self):
        """Test J Dilla tempo range."""
        min_bpm, max_bpm = get_preferred_tempo_range("J Dilla")
        assert min_bpm == 85
        assert max_bpm == 95

    def test_get_preferred_tempo_range_metro_boomin(self):
        """Test Metro Boomin tempo range."""
        min_bpm, max_bpm = get_preferred_tempo_range("Metro Boomin")
        assert min_bpm == 130
        assert max_bpm == 150

    def test_get_preferred_tempo_range_valid_range(self):
        """Test that tempo range is valid (min < max)."""
        for style in list_available_styles():
            min_bpm, max_bpm = get_preferred_tempo_range(style)
            assert min_bpm < max_bpm


class TestValidateTempoForStyle:
    """Tests for validate_tempo_for_style function."""

    def test_validate_tempo_in_range(self):
        """Test validating tempo within preferred range."""
        # J Dilla: 85-95 BPM
        assert validate_tempo_for_style("J Dilla", 90) is True
        assert validate_tempo_for_style("J Dilla", 85) is True
        assert validate_tempo_for_style("J Dilla", 95) is True

    def test_validate_tempo_out_of_range(self):
        """Test validating tempo outside preferred range."""
        # J Dilla: 85-95 BPM
        assert validate_tempo_for_style("J Dilla", 120) is False
        assert validate_tempo_for_style("J Dilla", 60) is False

    def test_validate_tempo_metro_boomin(self):
        """Test tempo validation for Metro Boomin."""
        # Metro Boomin: 130-150 BPM
        assert validate_tempo_for_style("Metro Boomin", 140) is True
        assert validate_tempo_for_style("Metro Boomin", 90) is False

    def test_validate_tempo_warn_only_default(self):
        """Test that warn_only defaults to True."""
        # Should not raise even if out of range
        result = validate_tempo_for_style("J Dilla", 200)
        assert result is False

    def test_validate_tempo_warn_only_false(self):
        """Test that warn_only=False raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_tempo_for_style("J Dilla", 200, warn_only=False)

        assert "outside preferred range" in str(exc_info.value)

    def test_validate_tempo_invalid_style(self):
        """Test validation with invalid style (should pass)."""
        # Invalid style returns True (validation passes)
        result = validate_tempo_for_style("Invalid Style", 120)
        assert result is True


class TestGetStyleDescription:
    """Tests for get_style_description function."""

    def test_get_style_description_returns_string(self):
        """Test that function returns a string."""
        desc = get_style_description("J Dilla")
        assert isinstance(desc, str)

    def test_get_style_description_not_empty(self):
        """Test that description is not empty."""
        desc = get_style_description("J Dilla")
        assert len(desc) > 0

    def test_get_style_description_j_dilla(self):
        """Test J Dilla description."""
        desc = get_style_description("J Dilla")
        assert "swing" in desc.lower() or "soulful" in desc.lower()

    def test_get_style_description_metro_boomin(self):
        """Test Metro Boomin description."""
        desc = get_style_description("Metro Boomin")
        assert "trap" in desc.lower() or "tight" in desc.lower()

    def test_get_style_description_different_for_each(self):
        """Test that descriptions are different for each style."""
        desc1 = get_style_description("J Dilla")
        desc2 = get_style_description("Metro Boomin")
        desc3 = get_style_description("Questlove")

        assert desc1 != desc2
        assert desc2 != desc3
        assert desc1 != desc3


class TestGetAllStylesInfo:
    """Tests for get_all_styles_info function."""

    def test_get_all_styles_info_returns_dict(self):
        """Test that function returns a dictionary."""
        info = get_all_styles_info()
        assert isinstance(info, dict)

    def test_get_all_styles_info_has_all_styles(self):
        """Test that all styles are included."""
        info = get_all_styles_info()
        assert "J Dilla" in info
        assert "Metro Boomin" in info
        assert "Questlove" in info

    def test_get_all_styles_info_structure(self):
        """Test that each style has complete information."""
        info = get_all_styles_info()

        for _style_name, style_data in info.items():
            assert "model_id" in style_data
            assert "model_path" in style_data
            assert "description" in style_data
            assert "preferred_tempo_range" in style_data
            assert "humanization" in style_data

    def test_get_all_styles_info_is_copy(self):
        """Test that returned dict is a copy (not reference)."""
        info1 = get_all_styles_info()
        info2 = get_all_styles_info()

        # Both calls should return the same data initially
        assert info1["J Dilla"]["description"] == info2["J Dilla"]["description"]

        # The function returns .copy(), which is a shallow copy
        # So modifying nested dicts will affect both (shallow copy behavior)
        # This is actually the expected behavior for .copy()
        # To test, we verify it's not the same object
        assert info1 is not info2


class TestStyleNotFoundError:
    """Tests for StyleNotFoundError exception."""

    def test_style_not_found_error_is_exception(self):
        """Test that StyleNotFoundError is an Exception."""
        assert issubclass(StyleNotFoundError, Exception)

    def test_style_not_found_error_message(self):
        """Test StyleNotFoundError with custom message."""
        error = StyleNotFoundError("Test error")
        assert str(error) == "Test error"

    def test_raising_style_not_found_error(self):
        """Test raising StyleNotFoundError."""
        with pytest.raises(StyleNotFoundError) as exc_info:
            raise StyleNotFoundError("Style not found")

        assert "not found" in str(exc_info.value).lower()
