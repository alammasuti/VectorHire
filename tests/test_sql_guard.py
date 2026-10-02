import unittest

from sqlalchemy import create_engine, text

from sql_guard import ReadOnlySQLDatabase, UnsafeSQLError, blocked_reason, validate_select


ALLOWED = [
    "SELECT Name, Skills FROM SoulsoftJobApplication WHERE Skills LIKE '%Python%'",
    "SELECT TOP 10 Name FROM SoulsoftJobApplication ORDER BY AppliedAt DESC;",
    "  select count(*) from SoulsoftJobApplication  ",
    "WITH c AS (SELECT Name, TotalExperience FROM SoulsoftJobApplication) SELECT * FROM c",
    # Keywords inside strings, comments and bracketed names are just data.
    "SELECT Name FROM SoulsoftJobApplication WHERE PositionAppliedFor = 'Delete; Drop Engineer'",
    "SELECT [Update] FROM SoulsoftJobApplication -- insert later",
    "SELECT Name FROM SoulsoftJobApplication WHERE Name = N'O''Brien'",
    "SELECT Name, UpdatedAt, CreatedBy FROM SoulsoftJobApplication",
]

BLOCKED = [
    "",
    "DELETE FROM SoulsoftJobApplication",
    "UPDATE SoulsoftJobApplication SET ExpectedSalary = 0",
    "DROP TABLE SoulsoftJobApplication",
    "INSERT INTO SoulsoftJobApplication (Name) VALUES ('x')",
    "TRUNCATE TABLE SoulsoftJobApplication",
    "EXEC xp_cmdshell 'dir'",
    "SELECT 1; DROP TABLE SoulsoftJobApplication",
    "SELECT 1 DELETE FROM SoulsoftJobApplication",  # T-SQL needs no semicolon
    "SELECT * INTO Copy FROM SoulsoftJobApplication",
    "WITH c AS (SELECT Id FROM SoulsoftJobApplication) DELETE FROM c",
    "SELECT * FROM OPENROWSET('SQLNCLI', 'x', 'SELECT 1')",
    "SELECT 1; WAITFOR DELAY '00:00:10'",
    "SELECT 1 /* unterminated",
    "SELECT 'unterminated",
    "SELECT Name FROM t; -- ok\nDROP TABLE t",
    "sp_who",
]


class ValidateSelectTests(unittest.TestCase):
    def test_allows_single_selects(self):
        for sql in ALLOWED:
            with self.subTest(sql=sql):
                self.assertEqual(validate_select(sql), sql)
                self.assertIsNone(blocked_reason(sql))

    def test_blocks_everything_else(self):
        for sql in BLOCKED:
            with self.subTest(sql=sql):
                with self.assertRaises(UnsafeSQLError) as ctx:
                    validate_select(sql)
                self.assertTrue(str(ctx.exception).startswith("Blocked query:"))
                self.assertIsNotNone(blocked_reason(sql))

    def test_no_sql_is_not_blocked(self):
        self.assertIsNone(blocked_reason(None))


class ReadOnlySQLDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as conn:
            conn.execute(text("CREATE TABLE candidates (Name TEXT)"))
            conn.execute(text("INSERT INTO candidates VALUES ('Asha')"))
        self.db = ReadOnlySQLDatabase(self.engine, include_tables=["candidates"])

    def test_select_runs(self):
        result, _ = self.db.run_sql("SELECT Name FROM candidates")
        self.assertIn("Asha", result)

    def test_write_never_reaches_database(self):
        with self.assertRaises(UnsafeSQLError):
            self.db.run_sql("DELETE FROM candidates")
        with self.engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM candidates")).scalar()
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
