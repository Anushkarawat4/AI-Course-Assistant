import os

import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException


def send_otp_email(email: str, otp: str):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = os.getenv("BREVO_API_KEY")

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        sender={
            "name": "CourseGPT",
            "email": "anukhu12@gmail.com",   # Your verified sender
        },
        to=[
            {
                "email": email
            }
        ],
        subject="CourseGPT OTP",
        text_content=f"Your CourseGPT OTP is: {otp}",
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
        print("Email sent:", response)
    except ApiException as e:
        print("Brevo Error:", e)
        raise