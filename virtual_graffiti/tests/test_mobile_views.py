from django.test import TestCase
from app.models import UserProfile, Laser
from virtual_graffiti.mobile_views import settings
import base64

class TestSettingsView(TestCase):
    def setUp(self):
        self.laser = Laser.objects.create(id='Re1d')
        self.user = UserProfile.objects.create(
            first_name="John", last_name="Doe", laser=self.laser
        )
        self.encoded_identifier = base64.b64encode(
            f"{self.user.first_name}_{self.user.last_name}_{self.user.laser.id}".encode(
                "utf-8"
            )
        ).decode("utf-8")

    def test_valid_user_identifier(self):
        # Simulate request with valid user identifier
        response = self.client.get(f"/settings/{self.encoded_identifier}")
        while response.status_code == 301:
            redirect_location = response.headers['Location']
            response = self.client.get(redirect_location)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["gradient"], True)
        self.assertEqual(response.context["from_gradient"], "#74EE15")
        self.assertEqual(response.context["to_gradient"], "#F000FF")
        self.assertEqual(response.context["first_name"], self.user.first_name)
        self.assertEqual(response.context["last_name"], self.user.last_name)
        self.assertEqual(response.context["laser_pointer"], self.user.laser.id)
        self.assertTemplateUsed(response, "settings.html")

    def test_invalid_user_identifier(self):
        # Simulate request with invalid user identifier
        invalid_identifier = "invalid_identifier"
        response = self.client.get(f"/settings/{invalid_identifier}")
        while response.status_code == 301:
            redirect_location = response.headers.get('Location')
            response = self.client.get(redirect_location)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/errors", response.headers.get('Location'))

    def test_missing_user(self):
        # Simulate request with a user identifier that doesn't exist
        altered_identifier = base64.b64encode(
            f"John_Doe_{str(self.laser.id) + str(1)}".encode("utf-8")
        ).decode("utf-8")
        response = self.client.get(f"/settings/{altered_identifier}")
        while response.status_code == 301:
            redirect_location = response.headers['Location']
            response = self.client.get(redirect_location)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/errors", response.headers.get('Location'))

    def test_missing_laser(self):
        # Simulate request with a user identifier that points to a non-existent laser
        altered_identifier = base64.b64encode(
            f"John_Doe_{1000}".encode("utf-8")
        ).decode("utf-8")
        response = self.client.get(f"/settings/{altered_identifier}")
        while response.status_code == 301:
            redirect_location = response.headers['Location']
            response = self.client.get(redirect_location)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/errors", response.headers.get('Location'))
