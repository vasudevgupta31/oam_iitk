import os
import threading
import json
import subprocess
from datetime import datetime
import configparser

from funcs.email import send_email_notification



def run_pipeline_in_background(experiment_name):
    """Run the pipeline process in a background thread that persists even if Streamlit disconnects"""

    memory_path = os.path.join('memory', experiment_name)
    config_path = os.path.join(memory_path, 'config.ini')
    config_file = configparser.ConfigParser()
    config_file.read(config_path)

    # Create a status file for this experiment
    status_file = os.path.join("memory", experiment_name, "status.json")
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    
    # Initialize status
    status = {
        "experiment": experiment_name,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "progress": 0,
        "current_stage": "Starting pipeline",
        "completed_stages": [],
        "logs": [],
        "error": None
    }

    # Save initial status
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    
    # Function to update status file
    def update_status(progress=None, stage=None, log=None, status_val=None, error=None):
        with open(status_file, "r") as f:
            current_status = json.load(f)
        
        if progress is not None:
            current_status["progress"] = progress
        
        if stage is not None:
            current_status["current_stage"] = stage
            if stage not in current_status["completed_stages"] and status_val == "completed":
                current_status["completed_stages"].append(stage)
        
        if log is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            current_status["logs"].append(f"{timestamp} - {log}")
        
        if status_val is not None:
            current_status["status"] = status_val
        
        if error is not None:
            current_status["error"] = error
            current_status["status"] = "failed"
        
        # Save updated status
        with open(status_file, "w") as f:
            json.dump(current_status, f, indent=2)
    
    def run_process():
        try:
            update_status(progress=5, stage="Initializing", log="Pipeline process started")
            
            # Build your command
            cmd = ["python", "main.py"]
            
            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Process output
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    # Update logs
                    update_status(log=line.strip())
                    
                    # Update progress based on pipeline stages
                    if "Data processing completed successfully" in line:
                        update_status(progress=25, stage="Data Processing", status_val="completed")
                    elif "Network training completed successfully" in line:
                        update_status(progress=50, stage="Model Training", status_val="completed")
                    elif "Sample generation completed" in line:
                        update_status(progress=75, stage="Sample Generation", status_val="completed")
                    elif "Novo analysis completed successfully" in line:
                        update_status(progress=95, stage="Analysis", status_val="completed")

            # Check return code
            return_code = process.poll()
            if return_code == 0:
                update_status(
                    progress=100, 
                    stage="Completion", 
                    status_val="completed",
                    log="🚀 Pipeline execution completed successfully!"
                )

                # Send email notification if configured
                if "Communication" in config_file and "email" in config_file["Communication"]:
                    email = config_file["Communication"]["email"]
                    send_email_notification(email, experiment_name, "Completed Successfully")
            else:
                update_status(
                    stage="Error", 
                    status_val="failed",
                    error=f"Process exited with code {return_code}",
                    log=f"❌ Pipeline execution failed with return code: {return_code}"
                )
                
                # Send failure email
                if "Communication" in config_file and "email" in config_file["Communication"]:
                    email = config_file["Communication"]["email"]
                    send_email_notification(email, experiment_name, "Failed")
                    
        except Exception as e:
            update_status(
                stage="Error",
                status_val="failed",
                error=str(e),
                log=f"❌ Error during pipeline execution: {str(e)}"
            )
            
            # Send failure email
            if "Communication" in config_file and "email" in config_file["Communication"]:
                email = config_file["Communication"]["email"]
                send_email_notification(email, experiment_name, "Failed")
    
    # Start the background thread
    thread = threading.Thread(target=run_process)
    thread.daemon = False  # Allow thread to continue even if main thread exits
    thread.start()
    
    return status_file
