"""
db_utils.py
-----------
Database helper functions for traffic management system
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime
from contextlib import contextmanager


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',          # Change to your MySQL username
    'password': 'Ssu@2005',          # Change to your MySQL password
    'database': 'traffic_management',
    'charset': 'utf8mb4'
}

# Helper: decide whether to return HTML (for regular form submits) or JSON (for AJAX)



@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    """
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        yield connection
    except Error as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()


def get_vehicle_owner(plate_number):
    """
    Get vehicle owner details by plate number
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    v.id as vehicle_id,
                    v.plate_number,
                    v.vehicle_type,
                    v.vehicle_model,
                    v.color,
                    o.id as owner_id,
                    o.name,
                    o.phone,
                    o.email,
                    o.address
                FROM vehicles v
                INNER JOIN vehicle_owners o ON v.owner_id = o.id
                WHERE v.plate_number = %s
            """
            
            cursor.execute(query, (plate_number,))
            result = cursor.fetchone()
            cursor.close()
            
            return result
            
    except Error as e:
        print(f"Error fetching vehicle owner: {e}")
        return None


def add_violation(plate_number, violation_data):
    """
    Add a new violation record
    
    violation_data should contain:
    - violation_type: str
    - fine_amount: float
    - location: str (optional)
    - image_path: str (optional)
    - plate_image_path: str (optional)
    - detected_text: str (optional)
    - confidence: float (optional)
    """
    try:
        # First check if vehicle exists
        owner_info = get_vehicle_owner(plate_number)
        owner_id = owner_info['owner_id'] if owner_info else None
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO violations 
                (plate_number, owner_id, violation_type, fine_amount, 
                 location, image_path, plate_image_path, detected_text, 
                 confidence, status, violation_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                plate_number,
                owner_id,
                violation_data.get('violation_type', 'No Helmet'),
                violation_data.get('fine_amount', 500.00),
                violation_data.get('location', 'Unknown'),
                violation_data.get('image_path'),
                violation_data.get('plate_image_path'),
                violation_data.get('detected_text'),
                violation_data.get('confidence'),
                'pending',
                datetime.now()
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            violation_id = cursor.lastrowid
            cursor.close()
            
            print(f"✅ Violation added: ID {violation_id}, Plate: {plate_number}")
            return violation_id
            
    except Error as e:
        print(f"Error adding violation: {e}")
        return None


def get_violations_by_plate(plate_number, status=None):
    """
    Get all violations for a plate number
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            if status:
                query = """
                    SELECT * FROM violations 
                    WHERE plate_number = %s AND status = %s
                    ORDER BY violation_date DESC
                """
                cursor.execute(query, (plate_number, status))
            else:
                query = """
                    SELECT * FROM violations 
                    WHERE plate_number = %s
                    ORDER BY violation_date DESC
                """
                cursor.execute(query, (plate_number,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
    except Error as e:
        print(f"Error fetching violations: {e}")
        return []


def get_all_pending_violations():
    """
    Get all pending violations
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    v.*,
                    o.name as owner_name,
                    o.phone as owner_phone,
                    ve.vehicle_type,
                    ve.vehicle_model
                FROM violations v
                LEFT JOIN vehicle_owners o ON v.owner_id = o.id
                LEFT JOIN vehicles ve ON v.plate_number = ve.plate_number
                WHERE v.status = 'pending'
                ORDER BY v.violation_date DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
    except Error as e:
        print(f"Error fetching pending violations: {e}")
        return []


def update_violation_status(violation_id, status, paid_date=None):
    """
    Update violation status (pending/paid/cancelled)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if paid_date:
                query = """
                    UPDATE violations 
                    SET status = %s, paid_date = %s
                    WHERE id = %s
                """
                cursor.execute(query, (status, paid_date, violation_id))
            else:
                query = """
                    UPDATE violations 
                    SET status = %s
                    WHERE id = %s
                """
                cursor.execute(query, (status, violation_id))
            
            conn.commit()
            cursor.close()
            
            print(f"✅ Violation {violation_id} status updated to: {status}")
            return True
            
    except Error as e:
        print(f"Error updating violation: {e}")
        return False


def get_owner_total_fines(owner_id):
    """
    Get total pending fines for an owner
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    COUNT(*) as violation_count,
                    SUM(fine_amount) as total_fines
                FROM violations
                WHERE owner_id = %s AND status = 'pending'
            """
            
            cursor.execute(query, (owner_id,))
            result = cursor.fetchone()
            cursor.close()
            
            return result
            
    except Error as e:
        print(f"Error fetching owner fines: {e}")
        return None


def search_vehicles(search_term):
    """
    Search vehicles by plate number or owner name
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    v.plate_number,
                    v.vehicle_type,
                    v.vehicle_model,
                    o.name as owner_name,
                    o.phone,
                    o.email
                FROM vehicles v
                INNER JOIN vehicle_owners o ON v.owner_id = o.id
                WHERE v.plate_number LIKE %s OR o.name LIKE %s
            """
            
            search_pattern = f"%{search_term}%"
            cursor.execute(query, (search_pattern, search_pattern))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
    except Error as e:
        print(f"Error searching vehicles: {e}")
        return []


def register_new_owner(owner_data):
    """
    Register a new vehicle owner
    
    owner_data should contain:
    - name, phone, email, address, aadhar_number, license_number
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if phone or email already exists
            check_query = "SELECT id FROM vehicle_owners WHERE phone = %s OR email = %s"
            cursor.execute(check_query, (owner_data['phone'], owner_data.get('email')))
            
            if cursor.fetchone():
                cursor.close()
                return {'success': False, 'error': 'Phone or email already registered'}
            
            query = """
                INSERT INTO vehicle_owners 
                (name, phone, email, address, aadhar_number, license_number)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            values = (
                owner_data['name'],
                owner_data['phone'],
                owner_data.get('email'),
                owner_data.get('address'),
                owner_data.get('aadhar_number'),
                owner_data.get('license_number')
            )
            
            cursor.execute(query, values)
            conn.commit()
            owner_id = cursor.lastrowid
            cursor.close()
            
            print(f"✅ Owner registered: ID {owner_id}, Name: {owner_data['name']}")
            return {'success': True, 'owner_id': owner_id}
            
    except Error as e:
        print(f"Error registering owner: {e}")
        return {'success': False, 'error': str(e)}


def register_new_vehicle(vehicle_data):
    """
    Register a new vehicle
    
    vehicle_data should contain:
    - plate_number, owner_id, vehicle_type, vehicle_brand, vehicle_model, color, etc.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if plate already exists
            check_query = "SELECT id FROM vehicles WHERE plate_number = %s"
            cursor.execute(check_query, (vehicle_data['plate_number'],))
            
            if cursor.fetchone():
                cursor.close()
                return {'success': False, 'error': 'Plate number already registered'}
            
            query = """
                INSERT INTO vehicles 
                (plate_number, owner_id, vehicle_type, vehicle_brand, vehicle_model, 
                 color, registration_date, chassis_number, engine_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                vehicle_data['plate_number'],
                vehicle_data['owner_id'],
                vehicle_data['vehicle_type'],
                vehicle_data.get('vehicle_brand'),
                vehicle_data.get('vehicle_model'),
                vehicle_data.get('color'),
                vehicle_data.get('registration_date'),
                vehicle_data.get('chassis_number'),
                vehicle_data.get('engine_number')
            )
            
            cursor.execute(query, values)
            conn.commit()
            vehicle_id = cursor.lastrowid
            cursor.close()
            
            print(f"✅ Vehicle registered: ID {vehicle_id}, Plate: {vehicle_data['plate_number']}")
            return {'success': True, 'vehicle_id': vehicle_id}
            
    except Error as e:
        print(f"Error registering vehicle: {e}")
        return {'success': False, 'error': str(e)}


def register_complete(owner_data, vehicle_data):
    """
    Register both owner and vehicle in one transaction
    """
    try:
        # First register owner
        owner_result = register_new_owner(owner_data)
        
        if not owner_result['success']:
            return owner_result
        
        # Then register vehicle with owner_id
        vehicle_data['owner_id'] = owner_result['owner_id']
        vehicle_result = register_new_vehicle(vehicle_data)
        
        if not vehicle_result['success']:
            return vehicle_result
        
        return {
            'success': True,
            'owner_id': owner_result['owner_id'],
            'vehicle_id': vehicle_result['vehicle_id']
        }
        
    except Exception as e:
        print(f"Error in complete registration: {e}")
        return {'success': False, 'error': str(e)}


def add_unregistered_violation(plate_number, violation_data):
    """
    Add violation for unregistered vehicle (plate not in system)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO unregistered_violations 
                (detected_plate, violation_type, fine_amount, location, 
                 image_path, plate_image_path, confidence, violation_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                plate_number,
                violation_data.get('violation_type', 'No Helmet'),
                violation_data.get('fine_amount', 500.00),
                violation_data.get('location', 'Unknown'),
                violation_data.get('image_path'),
                violation_data.get('plate_image_path'),
                violation_data.get('confidence'),
                datetime.now()
            )
            
            cursor.execute(query, values)
            conn.commit()
            violation_id = cursor.lastrowid
            cursor.close()
            
            print(f"⚠️ Unregistered violation added: ID {violation_id}, Plate: {plate_number}")
            return violation_id
            
    except Error as e:
        print(f"Error adding unregistered violation: {e}")
        return None


def get_unregistered_violations():
    """
    Get all unregistered violations (plates not in system)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT * FROM unregistered_violations
                WHERE resolved = FALSE
                ORDER BY violation_date DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
    except Error as e:
        print(f"Error fetching unregistered violations: {e}")
        return []


def link_unregistered_violation_to_vehicle(unregistered_id, vehicle_id):
    """
    Link an unregistered violation to a newly registered vehicle
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                UPDATE unregistered_violations
                SET resolved = TRUE, registered_vehicle_id = %s
                WHERE id = %s
            """
            
            cursor.execute(query, (vehicle_id, unregistered_id))
            conn.commit()
            cursor.close()
            
            return True
            
    except Error as e:
        print(f"Error linking violation: {e}")
        return False