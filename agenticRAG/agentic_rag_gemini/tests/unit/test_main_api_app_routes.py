import unittest

class TestMainApiRoutes(unittest.TestCase):
    def test_expected_routes_are_registered(self) -> None:
        try:
            from main_api import app
        except ModuleNotFoundError as exc:
            self.skipTest(f"Skipping route test because dependency is missing: {exc}")

        route_paths = {route.path for route in app.routes}
        expected = {
            "/answer",
            "/answer/status/{request_id}",
            "/query",
            "/process_query",
            "/tasks/{task_id}",
            "/health",
            "/info",
            "/sessions",
            "/sessions/{user_id}",
            "/sessions/{user_id}/{session_id}",
            "/sessions/{user_id}/{session_id}/summarize",
        }
        missing = expected - route_paths
        self.assertFalse(missing, f"Missing routes: {missing}")


if __name__ == "__main__":
    unittest.main()
