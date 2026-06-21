import os
import smtplib

from email.mime.text import MIMEText


def send_otp_email(email: str, otp: str):
    sender = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")

    msg = MIMEText(
        f"Your CourseGPT OTP is: {otp}"
    )

    msg["Subject"] = "CourseGPT OTP"
    msg["From"] = sender
    msg["To"] = email

    server = smtplib.SMTP(
        "smtp.gmail.com",
        587
    )

    server.starttls()

    server.login(
        sender,
        password
    )

    server.send_message(msg)

    server.quit()