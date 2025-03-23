import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



def send_email_notification(to_email, experiment_name, status, results=None):
    """
    Send email notification about experiment completion
    
    Parameters:
    to_email (str): Recipient email address
    experiment_name (str): Name of the experiment
    status (str): Success or failure
    results (dict, optional): Dictionary of result metrics
    """
    # Email server settings - you'll need to configure these
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    smtp_port = 587  # Standard port for TLS
    from_email = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")

    # Create message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"HGM Pipeline: Experiment '{experiment_name}' {status}"
    
    # Email body
    body = f"""
    <html>
    <body>
        <h2>HGM Pipeline Notification</h2>
        <p>Your experiment <b>{experiment_name}</b> has {status.lower()}.</p>
        
        <h3>Details:</h3>
        <ul>
            <li><b>Experiment:</b> {experiment_name}</li>
            <li><b>Status:</b> {status}</li>
            <li><b>Completion Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</li>
    """
    
    # Add results if available
    if results and isinstance(results, dict):
        body += "<h3>Results Summary:</h3><ul>"
        for key, value in results.items():
            body += f"<li><b>{key}:</b> {value}</li>"
        body += "</ul>"
    
    body += """
        </ul>
        <p>You can view detailed results in the Experiment Browser.</p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        print(f"Email notification sent to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False
