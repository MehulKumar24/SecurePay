"""Audit Logging System"""
import json
import os
from datetime import datetime
import hashlib


class AuditLogger:
    """Track all system actions for compliance and monitoring"""
    
    LOG_DIR = "audit_logs"
    
    @staticmethod
    def ensure_log_dir():
        """Create audit logs directory if it doesn't exist"""
        if not os.path.exists(AuditLogger.LOG_DIR):
            os.makedirs(AuditLogger.LOG_DIR)
    
    @staticmethod
    def log_action(action_type, details, user_id="system", session_id=None):
        """
        Log an action for audit trail
        
        Parameters:
        -----------
        action_type : str, type of action (upload, analyze, export, etc)
        details : dict, action details
        user_id : str, user performing action
        session_id : str, session identifier
        """
        AuditLogger.ensure_log_dir()
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'user_id': user_id,
            'session_id': session_id or 'unknown',
            'details': details,
            'details_hash': hashlib.sha256(json.dumps(details, default=str).encode()).hexdigest()[:16]
        }
        
        log_file = os.path.join(AuditLogger.LOG_DIR, f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            return True
        except Exception as e:
            print(f"Audit logging error: {str(e)}")
            return False
    
    @staticmethod
    def get_audit_history(action_type=None, days=7):
        """Retrieve audit logs"""
        AuditLogger.ensure_log_dir()
        logs = []
        
        try:
            for i in range(days):
                date_str = (datetime.now()).strftime('%Y%m%d')
                log_file = os.path.join(AuditLogger.LOG_DIR, f"audit_{date_str}.jsonl")
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            entry = json.loads(line)
                            if action_type is None or entry['action_type'] == action_type:
                                logs.append(entry)
        except Exception as e:
            print(f"Error retrieving audit logs: {str(e)}")
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    @staticmethod
    def get_compliance_report():
        """Generate compliance report from audit logs"""
        logs = AuditLogger.get_audit_history(days=30)
        
        report = {
            'total_actions': len(logs),
            'action_breakdown': {},
            'users_active': set(),
            'period': 'Last 30 days'
        }
        
        for log in logs:
            action = log['action_type']
            report['action_breakdown'][action] = report['action_breakdown'].get(action, 0) + 1
            report['users_active'].add(log['user_id'])
        
        report['users_active'] = list(report['users_active'])
        
        return report
