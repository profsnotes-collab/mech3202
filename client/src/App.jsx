import React, { useState, useEffect, useRef } from 'react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { auth, signOut } from './firebaseConfig'; 
import { onAuthStateChanged } from 'firebase/auth';
import styles from './index.module.css'; 
import SimpleLogin from './SimpleLogin'; 
import jokes from './jokes'; 

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [researchTopic, setResearchTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentJoke, setCurrentJoke] = useState("");
  const [data, setData] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const youtubeRef = useRef(null);

  const baseURL = 'https://profsnotes-ai.nn.r.appspot.com';

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setIsLoggedIn(!!user);
    });
    return unsubscribe;
  }, []);

  const handleLogout = () => {
    signOut(auth).then(() => {
      setIsLoggedIn(false);
      sessionStorage.removeItem('token');
    }).catch((error) => {
      console.error("Error signing out:", error);
    });
  };

  const fetchSuggestions = async () => {
    console.log('fetchSuggestions function called');

    if (researchTopic.length < 2) {
      setSuggestions([]);
      return;
    }
  
    try {
      console.log('Sending request to:', `${baseURL}/suggestions`);
      console.log('Request payload:', { query: researchTopic });
  
      const response = await fetch(`${baseURL}/suggestions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: researchTopic })
      });
  
      console.log('Response received:', response);
      console.log('Response status:', response.status);
  
      if (!response.ok) {
        console.error(`Error: ${response.status} - ${response.statusText}`);
        // Reading the response body as text can help identify if it's an error page or unexpected format
        const errorText = await response.text();
        console.error('Error details:', errorText);
        return; // Stop further processing
      }
  
      // Attempt to parse the response as JSON
      const suggestionsData = await response.json();
      console.log('Parsed JSON data:', suggestionsData);
      setSuggestions(suggestionsData);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };
  
  

  const fetchData = async () => {
    setLoading(true);
    setData(null);
    setCurrentJoke(jokes[Math.floor(Math.random() * jokes.length)]);

    try {
      const response = await fetch(`${baseURL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: researchTopic }),
      });

      if (response.ok) {
        const responseData = await response.json();
        setData(responseData);
      } else {
        throw new Error('Failed to fetch response');
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setData({ error: 'Error fetching data.' });
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    fetchData();
  };

  const highlightMatch = (text, query) => {
    const startIndex = text.toLowerCase().indexOf(query.toLowerCase());
    if (startIndex === -1) return text;
    const endIndex = startIndex + query.length;
    return (
      <>
        {text.slice(0, startIndex)}
        <strong>{text.slice(startIndex, endIndex)}</strong>
        {text.slice(endIndex)}
      </>
    );
  };

  const displaySuggestions = (suggestions) => {
    return suggestions.map((suggestion, index) => (
      <div key={index} className={styles.suggestionItem} onClick={() => setResearchTopic(suggestion)}>
        {highlightMatch(suggestion, researchTopic)}
      </div>
    ));
  };

  const renderProfsNotes = () => {
    if (!data?.notes_response?.response) {
      return <p>No ProfsNotes data available.</p>;
    }
  
    const { response } = data.notes_response;
  
    return (
      <div>
        {/* Display main text */}
        {response.Text && <p>{response.Text}</p>}
  
        {/* Display equations, images, and their descriptions */}
        {Object.entries(response).map(([key, value]) => {
          if (key.startsWith('Equation')) {
            const index = key.replace('Equation', '');
            const description = response[`EquationDescription${index}`];
            return (
              <div key={key}>
                <BlockMath math={value} />
                {description && <p>{description}</p>}
              </div>
            );
          } else if (key.startsWith('Image')) {
            const index = key.replace('Image', '');
            const description = response[`ImageDescription${index}`];
            return (
              <div key={key}>
                <img src={value} alt={`Image ${index}`} className={styles.image} />
                {description && <p>{description}</p>}
              </div>
            );
          }
          return null;
        })}
      </div>
    );
  };

  const renderWikipedia = () => {
    if (!data?.wikipedia_response?.results) {
      return <p>No Wikipedia data available.</p>;
    }
  
    const { Summary, "Top Three Images": topImages } = data.wikipedia_response.results;
  
    return (
      <div>
        {/* Display summary */}
        {Summary && <p>{Summary}</p>}
  
        {/* Display images and descriptions */}
        {topImages && Object.values(topImages).map((img, index) => (
          <div key={index}>
            <img src={img.URL} alt={`Wiki Image ${index}`} className={styles.image} />
            {img.Description && <p>{img.Description}</p>}
          </div>
        ))}
      </div>
    );
  };
  
  const renderYouTube = () => {
    if (!data?.youtube_response?.results) {
      return <p>No YouTube data available.</p>;
    }
  
    return (
      <div>
        {data.youtube_response.results.map((video, index) => {
          const url = new URL(video.timestamp);
          const videoId = url.searchParams.get('v');
          return (
            <div key={index}>
              <h3>{video.title || 'Untitled'}</h3>
              <p>{video.description || 'No description available'}</p>
              <iframe
                src={`https://www.youtube.com/embed/${videoId}?start=${url.searchParams.get('t')}`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title={`YouTube Video ${index}`}
                className={styles.youtubeVideo}
              ></iframe>
            </div>
          );
        })}
      </div>
    );
  };

  if (!isLoggedIn) {
    return <SimpleLogin onLogin={setIsLoggedIn} />;
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.menuToggle} onClick={() => setMenuOpen(!menuOpen)}>Menu</div>
        <div className={styles.title}>Professor Notes AI Assistant</div>
        <div className={styles.userSection} onClick={() => setUserMenuOpen(!userMenuOpen)}>
          User
          {userMenuOpen && (
            <div className={styles.userMenu}>
              <button className={styles.logoutButton} onClick={handleLogout}>Logout</button>
            </div>
          )}
        </div>
      </header>

      {menuOpen && (
        <nav className={styles.menu}>
          <button onClick={() => youtubeRef.current.scrollIntoView({ behavior: 'smooth' })}>YouTube Video</button>
          <button>Settings</button>
          <button className={styles.logoutButton} onClick={handleLogout}>Logout</button>
        </nav>
      )}

      <form className={styles.searchForm} onSubmit={handleSearch}>
        <input
          type="text"
          value={researchTopic}
          onChange={(e) => setResearchTopic(e.target.value)}
          onKeyUp={fetchSuggestions}
          placeholder="Enter your research topic"
        />
        <button type="submit">Search</button>
        {suggestions.length > 0 && (
          <div className={styles.suggestions}>
            {displaySuggestions(suggestions)}
          </div>
        )}
      </form>

      {loading ? (
        <div className={styles.loaderContainer}>
          <div className={styles.loader}></div>
          <p className={styles.joke}>{currentJoke}</p>
        </div>
      ) : (
        <main className={styles.content}>
          <section className={styles.section}>
            <h2>Prof Notes</h2>
            {renderProfsNotes()}
          </section>

          <section className={styles.section}>
            <h2>Wikipedia Summary</h2>
            {renderWikipedia()}
          </section>

          <section className={`${styles.section} ${styles.youtubeWindow}`} ref={youtubeRef}>
            <h2>YouTube Videos</h2>
            {renderYouTube()}
          </section>
        </main>
      )}

      <footer className={styles.footer}>
        <p>© ARIA 2024</p>
      </footer>
    </div>
  );
}

export default App;
