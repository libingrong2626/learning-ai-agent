from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.output_store import OutputStore
from paper_classifier.repositories.paper_repository import PaperRepository


class PaperRepositoryInputModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(self.id())
        self.logger.addHandler(logging.NullHandler())
        self.rows = pd.DataFrame([
            {
                "title": "same title",
                "abstract": "same abstract",
                "custom_field": "first",
                "source_id": 1,
            },
            {
                "title": "same title",
                "abstract": "same abstract",
                "custom_field": "duplicate",
                "source_id": 2,
            },
            {
                "title": "another title",
                "abstract": "another abstract",
                "custom_field": "other",
                "source_id": 3,
            },
        ])

    def test_xlsx_mode_reads_and_keeps_first_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.xlsx"
            self.rows.to_excel(input_path, index=False, engine="openpyxl")
            config = AppConfig(
                input_file_type="xlsx",
                xlsx_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(2, len(result))
        self.assertEqual("first", result.iloc[0]["custom_field"])
        self.assertIn("论文唯一标识", result.columns)

    def test_csv_mode_reads_utf8_and_uses_same_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            self.rows.to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(2, len(result))
        self.assertEqual("first", result.iloc[0]["custom_field"])
        self.assertIn("论文唯一标识", result.columns)

    def test_csv_mode_supports_gb18030(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            self.rows.to_csv(input_path, index=False, encoding="gb18030")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(2, len(result))

    def test_csv_mode_supports_utf16_with_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            self.rows.to_csv(input_path, index=False, encoding="utf-16")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(2, len(result))
        self.assertEqual("first", result.iloc[0]["custom_field"])

    def test_csv_mode_supports_utf16_without_bom(self) -> None:
        # Excel/WPS "Unicode 文本"另存会产出这种无 BOM 的 UTF-16 LE
        content = "title,abstract,custom_field,source_id\n" \
                  "same title,same abstract,first,1\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            input_path.write_bytes(content.encode("utf-16-le"))
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(1, len(result))
        self.assertEqual("first", result.iloc[0]["custom_field"])

    def test_csv_mode_supports_utf16_be_with_bom(self) -> None:
        content = "title,abstract,custom_field,source_id\n" \
                  "same title,same abstract,first,1\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            input_path.write_bytes(b"\xfe\xff" + content.encode("utf-16-be"))
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(1, len(result))

    def test_unrecognizable_encoding_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            input_path.write_bytes(
                b"id,title\n" + bytes(range(0x80, 0x90))  # 非法字节序列
            )
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            with self.assertRaisesRegex(ValueError, "编码无法识别"):
                PaperRepository(config, self.logger).load()

    def test_headers_are_found_ignoring_case_and_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            renamed = self.rows.rename(columns={
                "title": " Title ",
                "abstract": "ABSTRACT",
            })
            renamed.to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertIn("title", result.columns)
        self.assertIn("abstract", result.columns)
        self.assertEqual("same title", result.iloc[0]["title"])

    def test_classified_output_keeps_every_source_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "papers.csv"
            output_path = temp_path / "classified"
            self.rows.to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
                save_dir=output_path,
            )
            result = PaperRepository(config, self.logger).load()

            saved = OutputStore(config, self.logger).save_classified(
                result.iloc[[0]], {0: "management"}, set()
            )
            output = pd.read_csv(
                output_path / "management.csv", encoding="utf-8-sig"
            )

        self.assertEqual(1, len(saved))
        self.assertEqual(
            ["title", "abstract", "custom_field", "source_id"],
            list(output.columns),
        )
        self.assertEqual("first", output.iloc[0]["custom_field"])

    def test_append_to_legacy_output_keeps_title_and_abstract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "papers.csv"
            output_path = temp_path / "classified"
            output_path.mkdir()
            pd.DataFrame([{
                "标题": "old title",
                "摘要": "old abstract",
                "custom_field": "old",
                "source_id": 0,
            }]).to_csv(
                output_path / "management.csv",
                index=False,
                encoding="utf-8-sig",
            )
            self.rows.to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
                save_dir=output_path,
            )
            result = PaperRepository(config, self.logger).load()

            OutputStore(config, self.logger).save_classified(
                result.iloc[[0]], {0: "management"}, set()
            )
            output = pd.read_csv(
                output_path / "management.csv", encoding="utf-8-sig"
            )

        self.assertEqual("same title", output.iloc[1]["标题"])
        self.assertEqual("same abstract", output.iloc[1]["摘要"])

    def test_duplicate_non_required_headers_are_tolerated(self) -> None:
        """无关列（doi/DOI、md5/MD5）规范化后重复不应阻止读取。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            wide = pd.DataFrame([{
                "title": "t",
                "abstract": "a",
                "doi": "10.1/x",
                "DOI": "10.2/y",
                "md5": "aaa",
                "MD5": "bbb",
                "filename": "paper1.pdf",
            }])
            wide.to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            result = PaperRepository(config, self.logger).load()

        self.assertEqual(1, len(result))
        self.assertIn("DOI", result.columns)
        self.assertIn("MD5", result.columns)
        self.assertEqual("10.2/y", result.iloc[0]["DOI"])

    def test_duplicate_required_headers_still_raise(self) -> None:
        """必需列（title/Title）规范化后重复仍应报错。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "papers.csv"
            pd.DataFrame([{
                "title": "t1",
                "Title": "t2",
                "abstract": "a",
            }]).to_csv(input_path, index=False, encoding="utf-8-sig")
            config = AppConfig(
                input_file_type="csv",
                csv_input_path=input_path,
            )

            with self.assertRaisesRegex(ValueError, r"[Tt]itle"):
                PaperRepository(config, self.logger).load()

    def test_invalid_input_mode_is_rejected(self) -> None:
        config = AppConfig(input_file_type="json")

        with self.assertRaisesRegex(ValueError, "xlsx.*csv"):
            PaperRepository(config, self.logger).load()


class OutputStoreSaveOneTests(unittest.TestCase):
    """save_one 单篇立即写入：创建/追加/分学科文件/重复跳过。"""

    def _make_config(self, temp_dir: Path) -> AppConfig:
        return AppConfig(
            save_dir=Path(temp_dir) / "分类结果",
            title_column="title",
            abstract_column="abstract",
            filename_column="filename",
        )

    def _make_row(self, paper_id: str, title: str, abstract: str) -> pd.Series:
        return pd.Series(
            {
                "论文唯一标识": paper_id,
                "title": title,
                "abstract": abstract,
                "filename": f"{paper_id}.pdf",
            }
        )

    def test_first_save_creates_csv_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(self._make_config(Path(temp_dir)), logging.getLogger(self.id()))
            saved = store.save_one(
                self._make_row("p1", "标题一", "摘要一"), "数学", set()
            )
            save_path = Path(temp_dir) / "分类结果" / "数学.csv"

            self.assertTrue(saved)
            self.assertTrue(save_path.exists())
            df = pd.read_csv(save_path, encoding="utf-8-sig")
            self.assertEqual(1, len(df))
            self.assertEqual("p1.pdf", df.iloc[0]["filename"])
            self.assertEqual("标题一", df.iloc[0]["title"])

    def test_second_save_appends_without_duplicate_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(self._make_config(Path(temp_dir)), logging.getLogger(self.id()))
            already: set[str] = set()
            store.save_one(self._make_row("p1", "标题一", "摘要一"), "数学", already)
            saved = store.save_one(
                self._make_row("p2", "标题二", "摘要二"), "数学", already
            )
            save_path = Path(temp_dir) / "分类结果" / "数学.csv"

            self.assertTrue(saved)
            df = pd.read_csv(save_path, encoding="utf-8-sig")
            self.assertEqual(2, len(df))
            self.assertEqual(["p1.pdf", "p2.pdf"], df["filename"].tolist())

    def test_different_categories_write_separate_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(self._make_config(Path(temp_dir)), logging.getLogger(self.id()))
            already: set[str] = set()
            store.save_one(self._make_row("p1", "标题一", "摘要一"), "数学", already)
            store.save_one(self._make_row("p2", "标题二", "摘要二"), "经济学", already)

            math_df = pd.read_csv(
                Path(temp_dir) / "分类结果" / "数学.csv", encoding="utf-8-sig"
            )
            econ_df = pd.read_csv(
                Path(temp_dir) / "分类结果" / "经济学.csv", encoding="utf-8-sig"
            )
            self.assertEqual(["p1.pdf"], math_df["filename"].tolist())
            self.assertEqual(["p2.pdf"], econ_df["filename"].tolist())

    def test_duplicate_paper_id_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(self._make_config(Path(temp_dir)), logging.getLogger(self.id()))
            already: set[str] = set()
            store.save_one(self._make_row("p1", "标题一", "摘要一"), "数学", already)
            saved = store.save_one(
                self._make_row("p1", "标题一", "摘要一"), "数学", already
            )
            save_path = Path(temp_dir) / "分类结果" / "数学.csv"

            self.assertFalse(saved)
            df = pd.read_csv(save_path, encoding="utf-8-sig")
            self.assertEqual(1, len(df))


if __name__ == "__main__":
    unittest.main()
