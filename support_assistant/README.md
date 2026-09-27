\# Zepto Support Assistant



\## Overview



This module implements a small GenAI-style support assistant for Zepto policy questions.



The service uses:



\- 8 Zepto policy documents

\- Sentence Transformers (`all-MiniLM-L6-v2`) for local embeddings

\- ChromaDB for vector storage and retrieval

\- LangGraph for orchestration

\- Pydantic for structured responses

\- FastAPI for the `/ask` API

\- A deterministic `MOCK\_LLM` mode for offline execution



The default mode is `MOCK\_LLM=1`, so the application can be demonstrated without an API key.



\---



\## Architecture



The application follows this flow:



```text

User Query

&#x20;   |

&#x20;   v

FastAPI POST /ask

&#x20;   |

&#x20;   v

LangGraph StateGraph

&#x20;   |

&#x20;   v

classify\_intent

&#x20;   |

&#x20;   +------------------------------+

&#x20;   |                              |

&#x20;   | policy\_question              | general\_question

&#x20;   v                              v

retrieve\_and\_answer          direct\_answer

&#x20;   |                              |

&#x20;   v                              v

ChromaDB top-3 retrieval      Fixed mock response

&#x20;   |

&#x20;   v

Structured Pydantic response

&#x20;   |

&#x20;   v

FastAPI JSON response

