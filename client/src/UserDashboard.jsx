import React, { useState, useEffect } from 'react';
import { getAuth, onAuthStateChanged, updateProfile } from 'firebase/auth';
import { Link } from 'react-router-dom';
import styles from './UserDashboard.module.css';

const UserDashboard = () => {
  const [user, setUser] = useState(null);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [userClass, setUserClass] = useState('');

  useEffect(() => {
    const auth = getAuth();
    onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        setName(currentUser.displayName);
        setEmail(currentUser.email);
        // Fetch additional details like age, gender, class from your database if needed
      }
    });
  }, []);

  const handleEdit = () => {
    setEditing(true);
  };

  const handleSave = () => {
    const auth = getAuth();
    const currentUser = auth.currentUser;

    updateProfile(currentUser, {
      displayName: name,
    }).then(() => {
      // Update other details in your database if needed
      setEditing(false);
      setUser({
        ...currentUser,
        displayName: name,
        // Add other details like age, gender, class
      });
    }).catch((error) => {
      console.error('Error updating profile:', error);
    });
  };

  return (
    <div className={styles.dashboardContainer}>
      <h2>User Dashboard</h2>
      {user && (
        <div className={styles.userDetails}>
          <div>
            <strong>Name:</strong>
            {editing ? (
              <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
            ) : (
              <span>{name}</span>
            )}
          </div>
          <div>
            <strong>Email:</strong> <span>{email}</span>
          </div>
          <div>
            <strong>Age:</strong>
            {editing ? (
              <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />
            ) : (
              <span>{age}</span>
            )}
          </div>
          <div>
            <strong>Gender:</strong>
            {editing ? (
              <input type="text" value={gender} onChange={(e) => setGender(e.target.value)} />
            ) : (
              <span>{gender}</span>
            )}
          </div>
          <div>
            <strong>Class:</strong>
            {editing ? (
              <input type="text" value={userClass} onChange={(e) => setUserClass(e.target.value)} />
            ) : (
              <span>{userClass}</span>
            )}
          </div>
          <div>
            <strong>User ID:</strong> <span>{user.uid}</span>
          </div>
          {editing ? (
            <button className={styles.saveButton} onClick={handleSave}>Save</button>
          ) : (
            <button className={styles.editButton} onClick={handleEdit}>Edit</button>
          )}
        </div>
      )}
      <Link to="/" className={styles.homeButton}>Home</Link>
    </div>
  );
};

export default UserDashboard;
