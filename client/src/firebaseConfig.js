// src/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
import { getAuth, signOut as firebaseSignOut, onAuthStateChanged } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyC3BqxXlnoq_WOB3EeXppAJ2riMkRulg7c",
  authDomain: "profsnotes.firebaseapp.com",
  projectId: "profsnotes",
  storageBucket: "profsnotes.appspot.com",
  messagingSenderId: "443678693481",  
  appId: "1:443678693481:web:1e2ed6864d5a030330eaf8"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getFirestore(app);
const auth = getAuth(app);

export { auth, db, firebaseSignOut as signOut, onAuthStateChanged };
