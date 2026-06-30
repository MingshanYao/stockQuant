"""测试 concept 模块：个股板块/概念归属。"""

import pytest


class TestGetConceptBlocks:
    """get_concept_blocks 测试。"""

    def test_returns_expected_dict_structure(self):
        """返回 dict 包含 total/boards/tags。"""
        from stockquant.signals.concept import get_concept_blocks

        result = get_concept_blocks("600519")
        assert isinstance(result, dict)
        assert "total" in result
        assert "boards" in result
        assert "tags" in result
        assert isinstance(result["total"], int)
        assert isinstance(result["boards"], list)
        assert isinstance(result["tags"], list)

    def test_maotai_has_many_boards(self):
        """茅台应归属多个板块。"""
        from stockquant.signals.concept import get_concept_blocks

        result = get_concept_blocks("600519")
        assert result["total"] > 10, \
            f"茅台应有>10个板块归属, 实际{result['total']}"

    def test_boards_have_expected_fields(self):
        """每个板块包含 name/code/change_pct/lead_stock。"""
        from stockquant.signals.concept import get_concept_blocks

        result = get_concept_blocks("600519")
        if result["boards"]:
            board = result["boards"][0]
            for key in ("name", "code", "change_pct", "lead_stock"):
                assert key in board, f"缺少字段: {key}"

    def test_tags_equals_board_names(self):
        """tags 列表应与 boards 的 name 一一对应。"""
        from stockquant.signals.concept import get_concept_blocks

        result = get_concept_blocks("600519")
        board_names = [b["name"] for b in result["boards"]]
        assert result["tags"] == board_names

    @pytest.mark.parametrize("code_input", [
        "sh600519",
        "600519.SH",
        "SH600519",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sh 前缀和 .SH 后缀。"""
        from stockquant.signals.concept import get_concept_blocks

        result = get_concept_blocks(code_input)
        assert result["total"] > 0, f"代码 {code_input} 归一化后应有数据"
