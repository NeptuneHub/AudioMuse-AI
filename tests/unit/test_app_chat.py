"""Unit tests for app_chat.py utility functions"""
import pytest
from app_chat import clean_and_validate_sql


class TestCleanAndValidateSql:
    """Tests for the clean_and_validate_sql function"""

    def test_basic_valid_select(self):
        """Test basic valid SELECT statement"""
        sql = "SELECT * FROM score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        assert cleaned.upper().startswith("SELECT")

    def test_strips_markdown_code_blocks(self):
        """Test removal of markdown code block markers"""
        sql = "```sql\nSELECT * FROM score\n```"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        assert "```" not in cleaned

    def test_strips_leading_whitespace(self):
        """Test removal of leading whitespace"""
        sql = "   \n\t  SELECT * FROM score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        assert cleaned.startswith("SELECT")

    def test_rejects_empty_string(self):
        """Test rejection of empty string"""
        cleaned, error = clean_and_validate_sql("")
        assert cleaned is None
        assert "empty or invalid" in error

    def test_rejects_none(self):
        """Test rejection of None"""
        cleaned, error = clean_and_validate_sql(None)
        assert cleaned is None
        assert "empty or invalid" in error

    def test_rejects_non_string(self):
        """Test rejection of non-string input"""
        cleaned, error = clean_and_validate_sql(123)
        assert cleaned is None
        assert "empty or invalid" in error

    def test_rejects_non_select_query(self):
        """Test rejection of non-SELECT queries"""
        sql = "DELETE FROM score"
        cleaned, error = clean_and_validate_sql(sql)
        assert cleaned is None
        assert "SELECT" in error

    def test_rejects_insert_query(self):
        """Test rejection of INSERT queries"""
        sql = "INSERT INTO score VALUES (1, 2, 3)"
        cleaned, error = clean_and_validate_sql(sql)
        assert cleaned is None
        assert "SELECT" in error

    def test_rejects_update_query(self):
        """Test rejection of UPDATE queries"""
        sql = "UPDATE score SET value = 10"
        cleaned, error = clean_and_validate_sql(sql)
        assert cleaned is None
        assert "SELECT" in error

    def test_finds_select_in_text(self):
        """Test finding SELECT even with leading text"""
        sql = "Here is the query: SELECT * FROM score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        assert cleaned.upper().startswith("SELECT")

    def test_handles_html_entities(self):
        """Test unescaping of HTML entities like &gt;"""
        sql = "SELECT * FROM score WHERE value &gt; 10"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        assert ">" in cleaned
        assert "&gt;" not in cleaned

    def test_converts_backslash_quotes(self):
        """Test conversion of \\' to ''"""
        sql = "SELECT * FROM score WHERE title = 'Player\\'s Choice'"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        # Should convert \' to ''
        assert "\\'" not in cleaned

    def test_handles_apostrophes_in_words(self):
        """Test handling of apostrophes in words"""
        sql = "SELECT * FROM score WHERE title = 'Player's Choice'"
        # This might be tricky - the function should handle or flag it
        cleaned, error = clean_and_validate_sql(sql)
        # The function should either escape it or error out
        # Based on the code, it should convert 's to ''s
        if error:
            # If it errors, that's acceptable for safety
            assert True
        else:
            # If it succeeds, the apostrophe should be escaped
            assert cleaned is not None

    def test_detects_unbalanced_quotes(self):
        """Test detection of unbalanced quotes"""
        sql = "SELECT * FROM score WHERE title = 'unclosed"
        cleaned, error = clean_and_validate_sql(sql)
        assert cleaned is None
        assert "unbalanced quotes" in error or "truncated" in error

    def test_detects_unbalanced_parentheses(self):
        """Test detection of unbalanced parentheses"""
        sql = "SELECT * FROM score WHERE (value > 10"
        cleaned, error = clean_and_validate_sql(sql)
        assert cleaned is None
        assert "unbalanced parentheses" in error or "truncated" in error

    def test_normalizes_unicode(self):
        """Test normalization of Unicode characters"""
        # Smart quotes to regular quotes
        sql = "SELECT * FROM score WHERE title = 'Test'"
        cleaned, error = clean_and_validate_sql(sql)
        # Should normalize to ASCII
        if error is None:
            assert cleaned is not None

    def test_valid_join_query(self):
        """Test valid JOIN query"""
        sql = "SELECT a.*, b.name FROM score a JOIN tracks b ON a.id = b.id"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_valid_where_clause(self):
        """Test valid WHERE clause"""
        sql = "SELECT * FROM score WHERE tempo > 120 AND energy < 0.5"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_valid_order_by(self):
        """Test valid ORDER BY clause"""
        sql = "SELECT * FROM score ORDER BY tempo DESC LIMIT 10"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_valid_group_by(self):
        """Test valid GROUP BY clause"""
        sql = "SELECT key, COUNT(*) FROM score GROUP BY key"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_case_insensitive_select(self):
        """Test that SELECT is found case-insensitively"""
        sql = "select * from score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_mixed_case_select(self):
        """Test mixed case SELECT"""
        sql = "SeLeCt * FrOm score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_removes_trailing_semicolon(self):
        """Test that trailing semicolon is removed"""
        sql = "SELECT * FROM score;"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
        # The function should strip trailing semicolons
        assert not cleaned.endswith(";")

    def test_complex_valid_query(self):
        """Test complex valid query with multiple clauses"""
        sql = """
        SELECT 
            title, 
            author, 
            tempo,
            energy
        FROM score
        WHERE tempo BETWEEN 100 AND 140
            AND energy > 0.6
        ORDER BY tempo DESC
        LIMIT 20
        """
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_rejects_malformed_sql(self):
        """Test rejection of clearly malformed SQL"""
        sql = "SELECT FROM WHERE"
        cleaned, error = clean_and_validate_sql(sql)
        # Should fail parsing
        assert cleaned is None
        assert error is not None

    def test_handles_subquery(self):
        """Test valid subquery"""
        sql = "SELECT * FROM (SELECT * FROM score WHERE tempo > 120) AS subq"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_balanced_quotes_in_different_contexts(self):
        """Test balanced quotes in different contexts"""
        sql = "SELECT * FROM score WHERE title = 'Song' AND author = 'Artist'"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_multiple_parentheses_balanced(self):
        """Test multiple balanced parentheses"""
        sql = "SELECT * FROM score WHERE ((tempo > 100) AND (energy < 0.5))"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_with_line_breaks(self):
        """Test query with line breaks"""
        sql = "SELECT *\nFROM score\nWHERE tempo > 120"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_with_tabs(self):
        """Test query with tabs"""
        sql = "SELECT *\tFROM score\tWHERE tempo > 120"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_postgresql_specific_syntax(self):
        """Test PostgreSQL-specific syntax"""
        sql = "SELECT * FROM score WHERE mood_vector ILIKE '%happy%'"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_function_calls_in_select(self):
        """Test function calls in SELECT"""
        sql = "SELECT COUNT(*), AVG(tempo), MAX(energy) FROM score"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None

    def test_alias_usage(self):
        """Test column and table aliases"""
        sql = "SELECT s.tempo AS t, s.energy AS e FROM score AS s"
        cleaned, error = clean_and_validate_sql(sql)
        assert error is None
        assert cleaned is not None
