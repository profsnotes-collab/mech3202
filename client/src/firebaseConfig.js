// src/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
import { getAuth, signOut as firebaseSignOut, onAuthStateChanged } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyC0V3vPArL00c3-ax2k9v8J5MCiNGygNkw",
  authDomain: "profs-notes-mech3202.firebaseapp.com",
  projectId: "profs-notes-mech3202",
  storageBucket: "profs-notes-mech3202.appspot.com",
  messagingSenderId: "573968364158",
  appId: "1:573968364158:web:e6e3021f43a81b60bade2f",
  measurementId: "G-NKH9GYWBTX"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getFirestore(app);
const auth = getAuth(app);

export { auth, db, firebaseSignOut as signOut, onAuthStateChanged };
