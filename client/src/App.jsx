import React, { useState, useEffect, useRef } from 'react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { auth, signOut } from './firebaseConfig'; 
import { onAuthStateChanged } from 'firebase/auth';
import styles from './index.module.css'; 
import SimpleLogin from './SimpleLogin'; 
import jokes from './jokes'; 
import LoadingScreen from './LoadingScreen'; 

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [token, setToken] = useState(null);
  const [lastQueryCredits, setLastQueryCredits] = useState(null);
  const [remainingCredits, setRemainingCredits] = useState(null);
  const [lastQueryCached, setLastQueryCached] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const [isCached, setIsCached] = useState(false);
  const [researchTopic, setResearchTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentJoke, setCurrentJoke] = useState("");
  const [data, setData] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const youtubeRef = useRef(null);

  const baseURL = 'https://profsnotes.uk.r.appspot.com';

  const handleLogin = async (idToken) => {
    try {
      const response = await fetch(`${baseURL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ idToken })
      });
  
      if (!response.ok) {
        throw new Error('Login failed');
      }
  
      const data = await response.json();
      setToken(data.token);
      sessionStorage.setItem('token', data.token);
      setIsLoggedIn(true);
      setLastQueryCached(data.last_query_cached);
  
      // Start the warm-up process
      setIsWarmingUp(true);
      warmUpServices(data.token);
    } catch (error) {
      console.error('Login error:', error);
    }
  };
  
  const warmUpServices = async (token) => {
    try {
      await new Promise(resolve => setTimeout(resolve, 5000)); 

      const healthCheck = await fetch(`${baseURL}/health`);
      if (healthCheck.ok) {
        console.log('Services are ready');
      } else {
        console.log('Services may not be fully ready, but proceeding anyway');
      }

      await fetchUserCredits(token);
    } catch (error) {
      console.log('Warm-up process encountered an error:', error);
    } finally {
      setIsWarmingUp(false);
    }
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) {
        setIsLoggedIn(false);
        setToken(null);
        setRemainingCredits(null);
        setLastQueryCached(false);
        sessionStorage.removeItem('token');
      } else {
        const storedToken = sessionStorage.getItem('token');
        if (storedToken) {
          setToken(storedToken);
          setIsLoggedIn(true);
          fetchUserCredits(storedToken);
        }
      }
    });
  
    return () => unsubscribe();
  }, []);
  
  const fetchUserCredits = async (currentToken) => {
    try {
      console.log('Fetching user credits...');
      console.log('Token:', currentToken ? `${currentToken.substring(0, 10)}...` : 'No token');
  
      const response = await fetch(`${baseURL}/user_credits`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${currentToken}`,
          'Content-Type': 'application/json'
        }
      });
  
      console.log('Response status:', response.status);
  
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Failed to fetch user credits: ${response.status} ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log('User credits data:', data);
      setRemainingCredits(data.credits);
    } catch (error) {
      console.error('Error fetching user credits:', error);
    }
  };
  
  const handleLogout = () => {
    signOut(auth).then(() => {
      setIsLoggedIn(false);
      setToken(null);
      setRemainingCredits(null);
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
          'Content-Type': 'application/json'        },
        body: JSON.stringify({ query: researchTopic })
      });
  
      console.log('Response received:', response);
      console.log('Response status:', response.status);
  
      if (!response.ok) {
        console.error(`Error: ${response.status} - ${response.statusText}`);
        const errorText = await response.text();
        console.error('Error details:', errorText);
        return; 
      }
  
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
      if (!token) {
        console.error('No token available');
        setData({ error: 'No authentication token available. Please log in again.' });
        return;
      }
  
      if (!researchTopic.trim()) {
        console.error('No query provided');
        setData({ error: 'Please enter a research topic.' });
        return;
      }
  
      console.log('Sending request to:', `${baseURL}/query`);
      console.log('Request payload:', { query: researchTopic, token });
      console.log('Token:', token ? `${token.substring(0, 10)}...` : 'No token');
  
      const response = await fetch(`${baseURL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic, token }),
      });
  
      console.log('Response status:', response.status);
  
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Failed to fetch response: ${response.status} ${response.statusText}`);
      }
  
      const responseData = await response.json();
      console.log('Response data:', responseData);
  
      setData(responseData);
      setRemainingCredits(responseData.remaining_credits);
      setIsCached(responseData.is_cached);
  
      if (!responseData.is_cached) {
        setLastQueryCredits(responseData.credits_used);
      } else {
        setLastQueryCredits(null);
      }
      
    } catch (error) {
      console.error('Error fetching response:', error);
      setData({ error: 'Error fetching data: ' + error.message });
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

  if (!isLoggedIn) {
    return <SimpleLogin onLogin={handleLogin} />;
  }

  const renderProfsNotes = () => {
    if (!data?.notes_response?.response) {
      return <p>No ProfsNotes data available.</p>;
    }
  
    const { response } = data.notes_response;
  
    const renderContent = () => {
      const content = [];
      
      if (response.Text) {
        content.push(<p key="main-text">{response.Text}</p>);
      }
      
      Object.entries(response).forEach(([key, value]) => {
        if (key.startsWith('Equation') && !key.startsWith('EquationDescription')) {
          const index = key.replace('Equation', '');
          const description = response[`EquationDescription${index}`];
          content.push(
            <div key={`equation-${index}`}>
              <BlockMath math={value} />
              {description && <p>{description}</p>}
            </div>
          );
        } else if (key.startsWith('Image') && !key.startsWith('ImageDescription')) {
          const index = key.replace('Image', '');
          const description = response[`ImageDescription${index}`];
          if (value) {
            content.push(
              <div key={`image-${index}`}>
                <img src={value} alt={`Image ${index}`} className={styles.image} onError={(e) => { e.target.style.display = 'none' }} />
                {description && <p>{description}</p>}
              </div>
            );
          } else if (description) {
            content.push(<p key={`image-desc-${index}`}>{description}</p>);
          }
        }
      });
      
      return content;
    };
  
    return <div>{renderContent()}</div>;
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
    return <SimpleLogin onLogin={handleLogin} />;
  }

  return (
    <div className={styles.app}>
      {isWarmingUp && (
        <LoadingScreen 
          messages={[
            "Doing important stuff...", 
            "Take a deep breath or something, almost done...",
          ]}
          interval={3000}
        />
      )}
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
  
      {isLoggedIn && remainingCredits !== null && (
        <div className={styles.creditDisplay}>
          Remaining Credits: {remainingCredits}
          {isCached 
            ? " (Last query was cached)" 
            : lastQueryCredits !== null && lastQueryCredits !== undefined 
              ? ` (Last query used: ${lastQueryCredits})` 
              : lastQueryCached 
                ? " (Last query was cached)"  
                : ''}
        </div>
      )}
  
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
