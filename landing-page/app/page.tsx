'use client'

import { useState, useEffect } from 'react'
import { 
  Shield, 
  Zap, 
  Lock, 
  Code, 
  Check, 
  ArrowRight,
  Cpu,
  Globe,
  Terminal,
  Camera,
  CreditCard,
  FileCheck,
  FileText,
  Users,
  Brain,
  Eye,
  TrendingUp,
  AlertTriangle,
  Search,
  Loader2
} from 'lucide-react'

interface AnalysisResult {
  is_scam: boolean
  confidence: number
  truth_score: number
  classification: string
  classification_emoji: string
  recommendation: string
  manipulation_patterns: Array<{pattern: string, text: string, explanation: string}>
  logical_fallacies: Array<{fallacy: string, text: string, explanation: string}>
  issues: string[]
  positive_indicators: string[]
  analysis_time_ms: number
}

export default function Home() {
  const [email, setEmail] = useState('')
  const [activeTab, setActiveTab] = useState('security')
  const [scamText, setScamText] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [analysisError, setAnalysisError] = useState('')
  const [waitlistLoading, setWaitlistLoading] = useState(false)
  const [waitlistSuccess, setWaitlistSuccess] = useState(false)
  const [waitlistError, setWaitlistError] = useState('')
  
  const joinWaitlist = async () => {
    if (!email || !email.includes('@')) {
      setWaitlistError('Please enter a valid email')
      return
    }
    
    setWaitlistLoading(true)
    setWaitlistError('')
    
    try {
      const response = await fetch('https://nomadallm.nomadahealth.com/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })
      
      const data = await response.json()
      
      if (data.success) {
        setWaitlistSuccess(true)
        setEmail('')
      } else {
        setWaitlistError(data.message || 'Something went wrong')
      }
    } catch (error) {
      console.error('Waitlist error:', error)
      setWaitlistError('Network error. Please try again.')
    } finally {
      setWaitlistLoading(false)
    }
  }
  
  const analyzeForScam = async () => {
    if (!scamText.trim() || scamText.length < 10) {
      setAnalysisError('Please enter at least 10 characters')
      return
    }
    
    setAnalyzing(true)
    setAnalysisError('')
    setAnalysisResult(null)
    
    try {
      const response = await fetch('https://nomadallm.nomadahealth.com/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: scamText })
      })
      
      if (!response.ok) {
        throw new Error('Analysis failed')
      }
      
      const result = await response.json()
      setAnalysisResult(result)
    } catch (error) {
      setAnalysisError('Analysis failed. Please try again.')
    } finally {
      setAnalyzing(false)
    }
  }

  const features = [
    {
      icon: <Brain className="w-8 h-8 text-primary-500" />,
      title: 'Embedded Intelligence',
      description: 'Add a brain to any application. Not just chat - reasoning, pattern detection, decision making.'
    },
    {
      icon: <Shield className="w-8 h-8 text-primary-500" />,
      title: 'Privacy-First',
      description: 'All processing happens locally. Your data never leaves your device.'
    },
    {
      icon: <Zap className="w-8 h-8 text-primary-500" />,
      title: 'Real-Time Processing',
      description: 'No network calls. Instant responses for video streams, transactions, and live data.'
    },
    {
      icon: <Lock className="w-8 h-8 text-primary-500" />,
      title: 'Built-in Security',
      description: 'PII detection, fraud analysis, and compliance tools included.'
    },
    {
      icon: <Code className="w-8 h-8 text-primary-500" />,
      title: 'Universal SDK',
      description: 'Works with Python, Swift, JavaScript, and more. Integrated Orchestrator CLI bridges local edge intelligence with global development workflows. Built to work natively with GitHub Copilot extensions.'
    },
    {
      icon: <Globe className="w-8 h-8 text-primary-500" />,
      title: 'Works Offline',
      description: 'No internet required. Perfect for edge devices and secure environments.'
    },
  ]

  const useCases = {
    security: {
      title: 'Security Camera',
      subtitle: 'Raspberry Pi + Camera Module',
      description: 'A Raspberry Pi security camera that detects intruders and suspicious behavior without sending video to the cloud. Perfect for homes, warehouses, and retail.',
      code: `from nomadallm import NomadaLLM
from ultralytics import YOLO

class SecurityAgent:
    def __init__(self):
        self.llm = NomadaLLM(provider="local")
        self.yolo = YOLO("yolov8n-pose.pt")
        
    async def analyze_frame(self, frame):
        poses = self.yolo(frame)
        
        result = await self.llm.reason(
            context=poses,
            rules="Detect sequence: scanning, grasping, concealing",
            task="Return: NORMAL, SUSPICIOUS, or ALERT"
        )
        
        return result`
    },
    smartfarm: {
      title: 'Smart Agriculture',
      subtitle: 'Raspberry Pi + Sensors',
      description: 'Autonomous irrigation and crop monitoring. Works offline in rural areas with no internet. Raspberry Pi analyzes soil sensors and weather patterns locally.',
      code: `from nomadallm import NomadaLLM

class SmartFarmAgent:
    def __init__(self):
        self.llm = NomadaLLM(provider="embedded")
        
    async def analyze_conditions(self, sensors):
        result = await self.llm.reason(
            context=sensors,
            rules="Optimize water usage based on conditions",
            task="Decide: IRRIGATE_NOW, WAIT, or ALERT"
        )
        
        return result`
    },
    industrial: {
      title: 'Industrial Monitor',
      subtitle: 'Raspberry Pi + Vibration Sensors',
      description: 'Predictive maintenance for factory machinery. Raspberry Pi analyzes vibration patterns to predict failures before they happen. Data never leaves the factory floor.',
      code: `from nomadallm import NomadaLLM

class IndustrialMonitor:
    def __init__(self):
        self.llm = NomadaLLM(provider="embedded")
        
    async def analyze_machine(self, readings):
        result = await self.llm.reason(
            context=readings,
            rules="Detect anomalies in vibration and temp",
            task="Predict: HEALTHY, MAINTENANCE_SOON, or FAILURE_IMMINENT"
        )
        
        return result`
    },
    medical: {
      title: 'Medical Wearable Hub',
      subtitle: 'Raspberry Pi + Health Sensors',
      description: 'HIPAA-compliant health monitoring. Raspberry Pi collects data from wearables, analyzes locally, and only sends alerts. Patient data never leaves the device.',
      code: `from nomadallm import NomadaLLM

class MedicalHub:
    def __init__(self):
        self.llm = NomadaLLM(privacy_mode="healthcare")
        
    async def analyze_vitals(self, patient):
        result = await self.llm.reason(
            context=patient,
            rules="HIPAA compliant vital assessment",
            task="Assess: NORMAL, MONITOR, or ALERT_DOCTOR"
        )
        
        return result`
    },
    pos: {
      title: 'POS Fraud Detection',
      subtitle: 'Raspberry Pi + Payment Terminal',
      description: 'Real-time fraud detection at point of sale. Raspberry Pi analyzes transaction patterns locally with zero latency. Perfect for retail stores and restaurants.',
      code: `from nomadallm import NomadaLLM

class POSFraudDetector:
    def __init__(self):
        self.llm = NomadaLLM(privacy_mode="banking")
        
    async def analyze_transaction(self, tx):
        result = await self.llm.reason(
            context=tx,
            rules="Detect fraud patterns in real-time",
            task="Decide: APPROVE, REVIEW, or BLOCK"
        )
        
        return result`
    }
  }

  const pricingTiers = [
    {
      name: 'Free',
      price: 0,
      period: 'forever',
      description: 'Perfect for testing and personal projects',
      features: [
        '100 calls/day',
        'All features included',
        'LLM Inference',
        'PII Detection',
        'Fraud Detection',
        'Fine-tuning API',
        'Community support'
      ],
      cta: 'Get Started Free',
      popular: false
    },
    {
      name: 'Indie',
      price: 9,
      period: '/month',
      description: 'For indie developers and small projects',
      features: [
        '10,000 calls/day',
        'All features included',
        'LLM Inference',
        'PII Detection',
        'Fraud Detection',
        'Fine-tuning API',
        'Email support'
      ],
      cta: 'Start Indie',
      popular: false
    },
    {
      name: 'Pro',
      price: 29,
      period: '/month',
      description: 'For growing applications',
      features: [
        '100,000 calls/day',
        'All features included',
        'LLM Inference',
        'PII Detection',
        'Fraud Detection',
        'Fine-tuning API',
        'Priority support'
      ],
      cta: 'Go Pro',
      popular: true
    },
    {
      name: 'Enterprise',
      price: 99,
      period: '/month',
      description: 'For large-scale deployments',
      features: [
        'Unlimited calls',
        'All features included',
        'LLM Inference',
        'PII Detection',
        'Fraud Detection',
        'Fine-tuning API',
        'SLA guarantee',
        'Dedicated support',
        'Custom deployment and private cluster orchestration'
      ],
      cta: 'Contact Sales',
      popular: false
    }
  ]

  const codeExample = `pip install nomadallm
nomada orchestrate --init

from nomadallm import NomadaLLM

# Initialize (free tier: 100 calls/day, all features)
brain = NomadaLLM()

# Agentic reasoning, not chat
result = await brain.reason(
    context=sensor_data,
    rules=business_rules,
    task="Analyze patterns and recommend action"
)`

  return (
    <main className="min-h-screen">
      {/* Developer Banner */}
      <div className="bg-primary-600 text-white py-3 px-4 text-center">
        <p className="text-sm md:text-base font-medium">
          For Developers: NomadaLLM Edge AI SDK is coming.{' '}
          <a href="#early-access" className="underline font-bold hover:text-primary-200">
            Join Early Access →
          </a>
        </p>
      </div>

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-slate-900 to-slate-800 text-white">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10"></div>
        <div className="container mx-auto px-6 py-24 relative">
          <div className="max-w-4xl mx-auto text-center">
            
            <p className="text-slate-400 text-sm mb-8">
              🍓 Runs on Raspberry Pi 5. <span className="text-primary-400 font-semibold">Edge AI for the Internet of Consciousness.</span>
            </p>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-4">
              <span className="gradient-text">Autonomous Intelligence</span> for the Edge
            </h1>
            
            <p className="text-xl text-slate-300 mb-6 max-w-2xl mx-auto">
              Edge AI SDK for Raspberry Pi and IoT.
              <span className="text-primary-400 font-medium"> No cloud. No latency. Total privacy.</span>
            </p>
            
            <p className="text-sm text-slate-400 mb-4">
              Fully compatible with GitHub Copilot SDK and MCP standards.
            </p>
            
            <p className="text-lg text-white font-medium mb-8 max-w-2xl mx-auto">
              Run AI on Raspberry Pi, edge devices, and embedded systems.<br />
              <span className="text-primary-400">100% Local. Works Offline. Zero Data Leakage.</span>
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <a href="#early-access" className="btn-primary inline-flex items-center gap-2">
                Join Early Access <ArrowRight className="w-5 h-5" />
              </a>
              <a href="#use-cases" className="btn-secondary">
                See Use Cases
              </a>
            </div>
            
            {/* Code Preview */}
            <div className="code-block text-left max-w-2xl mx-auto mb-12">
              <div className="flex items-center gap-2 mb-4">
                <Terminal className="w-5 h-5 text-primary-400" />
                <span className="text-primary-400 text-sm">Quick Start</span>
              </div>
              <pre><code>{codeExample}</code></pre>
            </div>
            
            {/* Performance Guarantee */}
            <div className="grid md:grid-cols-3 gap-6 max-w-3xl mx-auto">
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                <p className="text-primary-400 font-bold text-2xl mb-1">🍓 Pi 5 Ready</p>
                <p className="text-slate-400 text-sm">Runs on Raspberry Pi 5 (8GB)</p>
              </div>
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                <p className="text-primary-400 font-bold text-2xl mb-1">&lt; 50ms</p>
                <p className="text-slate-400 text-sm">Real-time edge inference</p>
              </div>
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                <p className="text-primary-400 font-bold text-2xl mb-1">Zero Cloud</p>
                <p className="text-slate-400 text-sm">No internet required</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FREE Scam Detector Tool */}
      <section className="py-16 bg-gradient-to-b from-slate-800 to-slate-900" id="scam-detector">
        <div className="container mx-auto px-6">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <div className="inline-flex items-center gap-2 bg-red-500/20 text-red-300 px-4 py-2 rounded-full text-sm mb-4">
                <Shield className="w-4 h-4" />
                FREE Tool - Try NomadaLLM Now
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Scam Detector
              </h2>
              <p className="text-slate-300">
                Paste any message, email, or conversation. We&apos;ll detect manipulation patterns, 
                logical fallacies, and scam indicators instantly.
              </p>
            </div>
            
            <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700">
              <textarea
                value={scamText}
                onChange={(e) => setScamText(e.target.value)}
                placeholder="Paste suspicious text here... 

Example: 'URGENT! You've won $1,000,000! Act now before it expires! Click here to claim your prize. Don't miss this limited time offer!'"
                className="w-full h-40 bg-slate-900 text-white rounded-lg p-4 border border-slate-600 focus:border-primary-500 focus:outline-none resize-none"
              />
              
              <div className="flex items-center justify-between mt-4">
                <span className="text-slate-400 text-sm">
                  {scamText.length}/10,000 characters
                </span>
                <button
                  onClick={analyzeForScam}
                  disabled={analyzing || scamText.length < 10}
                  className="btn-primary inline-flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Analyze for Scam
                    </>
                  )}
                </button>
              </div>
              
              {analysisError && (
                <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                  {analysisError}
                </div>
              )}
              
              {analysisResult && (
                <div className="mt-6 space-y-4">
                  {/* Main Result - Strict Thresholds: 0-59 HIGH RISK, 60-84 SUSPICIOUS, 85-100 SAFE */}
                  {(() => {
                    const score = analysisResult.truth_score;
                    const hasIssues = analysisResult.issues.length > 0 || analysisResult.manipulation_patterns.length > 0;
                    // Force HIGH RISK if score < 60 OR if any issues detected
                    const isHighRisk = score < 60 || hasIssues;
                    const isSuspicious = !isHighRisk && score < 85;
                    const isSafe = !isHighRisk && !isSuspicious;
                    
                    let bgClass = 'bg-green-500/20 border border-green-500/50';
                    let textClass = 'text-green-300';
                    let label = '✅ VERIFIED SAFE';
                    let emoji = '✅';
                    
                    if (isHighRisk) {
                      bgClass = 'bg-red-500/20 border border-red-500/50';
                      textClass = 'text-red-300';
                      label = '🚨 HIGH RISK - LIKELY SCAM';
                      emoji = '🚨';
                    } else if (isSuspicious) {
                      bgClass = 'bg-yellow-500/20 border border-yellow-500/50';
                      textClass = 'text-yellow-300';
                      label = '⚠️ SUSPICIOUS - PROCEED WITH CAUTION';
                      emoji = '⚠️';
                    }
                    
                    return (
                      <div className={`p-6 rounded-xl ${bgClass}`}>
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-4xl">{emoji}</span>
                          <div>
                            <h3 className={`text-2xl font-bold ${textClass}`}>
                              {label}
                            </h3>
                            <p className="text-slate-300">
                              Truth Score: {analysisResult.truth_score}/100 • 
                              Confidence: {Math.round(analysisResult.confidence * 100)}%
                            </p>
                          </div>
                        </div>
                        <p className="text-slate-200 mt-2">{analysisResult.recommendation}</p>
                      </div>
                    );
                  })()}
                  
                  {/* Issues Found */}
                  {analysisResult.issues.length > 0 && (
                    <div className="p-4 bg-slate-700/50 rounded-lg">
                      <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-yellow-400" />
                        Issues Detected ({analysisResult.issues.length})
                      </h4>
                      <ul className="space-y-1">
                        {analysisResult.issues.map((issue, i) => (
                          <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                            <span className="text-red-400">•</span>
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Manipulation Patterns */}
                  {analysisResult.manipulation_patterns.length > 0 && (
                    <div className="p-4 bg-slate-700/50 rounded-lg">
                      <h4 className="text-white font-semibold mb-2">
                        🎭 Manipulation Patterns
                      </h4>
                      <div className="space-y-2">
                        {analysisResult.manipulation_patterns.map((p, i) => (
                          <div key={i} className="text-sm">
                            <span className="text-red-400 font-medium">{p.pattern}:</span>
                            <span className="text-slate-300 ml-2">{p.explanation}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Positive Indicators */}
                  {analysisResult.positive_indicators.length > 0 && (
                    <div className="p-4 bg-slate-700/50 rounded-lg">
                      <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                        <Check className="w-5 h-5 text-green-400" />
                        Positive Indicators
                      </h4>
                      <ul className="space-y-1">
                        {analysisResult.positive_indicators.map((pos, i) => (
                          <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            {pos}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <p className="text-slate-500 text-xs text-center">
                    Analyzed in {analysisResult.analysis_time_ms.toFixed(0)}ms • Powered by NomadaLLM + NomadaGuard
                  </p>
                  
                  {/* Upsell CTA - Show after HIGH RISK detection */}
                  {(analysisResult.is_scam || analysisResult.truth_score < 60) && (
                    <div className="mt-6 p-6 bg-gradient-to-r from-primary-600/20 to-primary-500/20 border border-primary-500/50 rounded-xl">
                      <h4 className="text-white font-bold text-lg mb-2">
                        🛡️ Protect Your Business From Scams
                      </h4>
                      <p className="text-slate-300 text-sm mb-4">
                        Your analysis detected a <span className="text-red-400 font-semibold">Score of {analysisResult.truth_score}/100 (High Risk)</span>. 
                        Integrate this same detector into your own platform with just 3 lines of code.
                      </p>
                      <div className="flex flex-col sm:flex-row gap-3">
                        <a 
                          href="#pricing" 
                          className="btn-primary text-center flex-1"
                        >
                          Get SDK - From $9/mo
                        </a>
                        <a 
                          href="mailto:joaquin@nomadahealth.com?subject=NomadaLLM SDK Demo" 
                          className="btn-secondary text-center flex-1"
                        >
                          Request Demo
                        </a>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            <p className="text-center text-slate-400 text-sm mt-4">
              Want to build your own scam detector? <a href="#pricing" className="text-primary-400 hover:underline">Get the SDK →</a>
            </p>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="py-12 bg-white border-b">
        <div className="container mx-auto px-6">
          <p className="text-center text-slate-500 mb-8">Edge AI for every industry</p>
          <div className="flex flex-wrap justify-center items-center gap-12 opacity-60">
            <span className="text-2xl font-bold text-slate-400">🍓 Raspberry Pi</span>
            <span className="text-2xl font-bold text-slate-400">Industrial IoT</span>
            <span className="text-2xl font-bold text-slate-400">Healthcare</span>
            <span className="text-2xl font-bold text-slate-400">Agriculture</span>
            <span className="text-2xl font-bold text-slate-400">Retail</span>
          </div>
        </div>
      </section>

      {/* Use Cases Section - NEW */}
      <section className="py-24 bg-white" id="use-cases">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Beyond Chat: Real Intelligence Agents</h2>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              99% of LLM usage is chat. We built NomadaLLM for the other 1%. Embedded intelligence that makes decisions, detects patterns, and reasons about data.
            </p>
          </div>
          
          {/* Tab Navigation */}
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            {Object.entries(useCases).map(([key, useCase]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  activeTab === key 
                    ? 'bg-primary-500 text-white' 
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {useCase.title}
              </button>
            ))}
          </div>
          
          {/* Active Use Case */}
          <div className="max-w-4xl mx-auto">
            <div className="bg-slate-900 rounded-2xl p-8 text-white">
              <div className="flex items-center gap-3 mb-2">
                {activeTab === 'security' && <Camera className="w-6 h-6 text-primary-400" />}
                {activeTab === 'fraud' && <CreditCard className="w-6 h-6 text-primary-400" />}
                {activeTab === 'tax' && <FileText className="w-6 h-6 text-primary-400" />}
                {activeTab === 'hr' && <Users className="w-6 h-6 text-primary-400" />}
                {activeTab === 'marketing' && <TrendingUp className="w-6 h-6 text-primary-400" />}
                <h3 className="text-2xl font-bold">{useCases[activeTab as keyof typeof useCases].title}</h3>
              </div>
              <p className="text-primary-300 mb-4">{useCases[activeTab as keyof typeof useCases].subtitle}</p>
              <p className="text-slate-300 mb-6">{useCases[activeTab as keyof typeof useCases].description}</p>
              
              <div className="code-block">
                <pre><code>{useCases[activeTab as keyof typeof useCases].code}</code></pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Local vs Cloud Comparison */}
      <section className="py-24 bg-white" id="comparison">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Local vs Cloud AI</h2>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              Why enterprises are moving to local intelligence
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-slate-100">
                  <th className="text-left p-4 font-semibold">Feature</th>
                  <th className="text-left p-4 font-semibold text-primary-600">NomadaLLM (Local)</th>
                  <th className="text-left p-4 font-semibold text-slate-500">Standard Cloud AI</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-4 font-medium">Latency</td>
                  <td className="p-4 text-primary-600">&lt; 50ms (Real-time edge inference)</td>
                  <td className="p-4 text-slate-500">1s to 3s (Network dependent)</td>
                </tr>
                <tr className="border-b bg-slate-50">
                  <td className="p-4 font-medium">Data Privacy</td>
                  <td className="p-4 text-primary-600">Absolute (On-device)</td>
                  <td className="p-4 text-slate-500">Vulnerable (3rd party servers)</td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">Operational Cost</td>
                  <td className="p-4 text-primary-600">Fixed Subscription</td>
                  <td className="p-4 text-slate-500">Variable / Pay-per-token</td>
                </tr>
                <tr className="border-b bg-slate-50">
                  <td className="p-4 font-medium">Offline Ability</td>
                  <td className="p-4 text-primary-600">100% Functional</td>
                  <td className="p-4 text-slate-500">Zero</td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">Compliance</td>
                  <td className="p-4 text-primary-600">No BAA required. Data never leaves physical premises.</td>
                  <td className="p-4 text-slate-500">Requires BAA / DPA</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Compliance Section */}
      <section className="py-24 bg-slate-900 text-white" id="compliance">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Enterprise Trust & Compliance</h2>
            <p className="text-xl text-slate-300 max-w-2xl mx-auto">
              Built for Hospitals, Banks, and Air-Gapped Environments
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center p-6 bg-slate-800 rounded-xl border border-slate-700">
              <Shield className="w-12 h-12 text-primary-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Air-Gap Certified</h3>
              <p className="text-slate-400">Works in environments with NO internet. Zero external dependencies.</p>
            </div>
            
            <div className="text-center p-6 bg-slate-800 rounded-xl border border-slate-700">
              <Lock className="w-12 h-12 text-primary-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Data Sovereignty</h3>
              <p className="text-slate-400">Your data never leaves the hardware. Period. Full HIPAA/GDPR compliance.</p>
            </div>
            
            <div className="text-center p-6 bg-slate-800 rounded-xl border border-slate-700">
              <FileCheck className="w-12 h-12 text-primary-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Audit Ready</h3>
              <p className="text-slate-400">Simplified compliance documentation. No third-party data processors.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Sea Captain AI - Empathetic Edge Intelligence */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold mb-4">Strategic Pilot Persona</h2>
              <p className="text-xl text-slate-600">
                A proof-of-concept for high-reasoning agents in isolated environments.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 text-white">
              <div className="flex items-start gap-6">
                <div className="text-6xl">🧭</div>
                <div>
                  <h3 className="text-2xl font-bold mb-2">Strategic Pilot</h3>
                  <p className="text-primary-300 mb-4">High-reasoning agent with stoic decision-making framework</p>
                  <p className="text-slate-300 mb-6">
                    Built with a stoic decision-making framework. Running 100% offline on edge devices. 
                    No cloud. No tracking. Demonstrates that local AI can reason autonomously in isolated environments.
                  </p>
                  <div className="code-block text-sm">
                    <pre><code>{`pilot = NomadaLLM(persona="strategic_pilot")
decision = pilot.reason(
    context="System anomaly detected",
    rules="Stoic framework: focus on controllables",
    task="Recommend action"
)`}</code></pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-slate-50" id="features">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Why NomadaLLM?</h2>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              The only LLM SDK built from the ground up for privacy, security, and portability.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-slate-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-xl text-slate-600">Three steps to Embedded Intelligence</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary-600">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Install the Brain</h3>
              <p className="text-slate-600">pip install nomadallm</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary-600">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Initialize the Orchestrator</h3>
              <p className="text-slate-600">llm = NomadaLLM(mode="local")</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary-600">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Deploy Reasoning</h3>
              <p className="text-slate-600 mb-2">Stop sending messages; start processing patterns</p>
              <p className="text-primary-600 font-mono text-sm">result = llm.reason(context, rules, task)</p>
            </div>
          </div>
          
          {/* Full code example below the 3 steps */}
          <div className="mt-12 max-w-2xl mx-auto">
            <div className="code-block text-sm text-left">
              <pre><code>{`# Instead of a chat, show a DECISION:
result = await llm.reason(
    context=sensor_data, 
    rules=business_logic,
    task="Detect anomalies and execute trigger"
)`}</code></pre>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="py-24 bg-slate-50" id="pricing">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Simple, Transparent Pricing</h2>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              All features included in every tier. Only usage limits differ.
              <br />
              <span className="text-primary-600 font-semibold">Like Stripe and Twilio - test everything before you pay.</span>
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
            {pricingTiers.map((tier, index) => (
              <div 
                key={index} 
                className={`pricing-card ${tier.popular ? 'popular' : ''}`}
              >
                {tier.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-primary-500 text-white px-4 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                )}
                
                <h3 className="text-2xl font-bold mb-2">{tier.name}</h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold">${tier.price}</span>
                  <span className="text-slate-500">{tier.period}</span>
                </div>
                <p className="text-slate-600 mb-6">{tier.description}</p>
                
                <ul className="space-y-3 mb-8">
                  {tier.features.map((feature, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                      <span className="text-slate-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <a href="#early-access" className={tier.popular ? 'btn-primary w-full text-center block' : 'btn-secondary w-full text-center block'}>
                  {tier.price === 0 ? 'Try Free Demo' : 'Get Early Access'}
                </a>
                {tier.price > 0 && (
                  <p className="text-xs text-slate-500 text-center mt-2">
                    Available Now
                  </p>
                )}
              </div>
            ))}
          </div>
          
          {/* Custom Architecture CTA */}
          <div className="mt-12 text-center">
            <div className="bg-gradient-to-r from-primary-500/10 to-primary-600/10 border border-primary-500/30 rounded-2xl p-8 max-w-2xl mx-auto">
              <h3 className="text-2xl font-bold mb-2">Need Custom Architecture?</h3>
              <p className="text-slate-600 mb-6">
                Enterprise deployments, custom integrations, and dedicated support from the creator.
              </p>
              <a href="mailto:joaquin@nomadahealth.com?subject=NomadaLLM Custom Architecture" className="btn-primary inline-flex items-center gap-2">
                Contact for Custom Solutions <ArrowRight className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Early Access Section */}
      <section className="py-24 bg-white" id="early-access">
        <div className="container mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-6">FREE Tool - Try NomadaLLM Now</h2>
            <p className="text-xl text-slate-700 mb-4 leading-relaxed">
              Experience AI that never leaves your device. Try our <span className="text-primary-600 font-semibold">live scam detector demo</span> today.
            </p>
            <p className="text-lg text-slate-600 mb-8">
              Get early access to the full SDK.
            </p>
            
            <div className="max-w-md mx-auto">
              {waitlistSuccess ? (
                <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                  <div className="text-4xl mb-3">🎉</div>
                  <h3 className="text-lg font-semibold text-green-800 mb-2">You&apos;re on the list!</h3>
                  <p className="text-green-700">Check your email for confirmation.</p>
                </div>
              ) : (
                <>
                  <div className="flex gap-2">
                    <input 
                      type="email" 
                      placeholder="your@email.com"
                      value={email}
                      onChange={(e) => {
                        setEmail(e.target.value)
                        setWaitlistError('')
                      }}
                      onKeyDown={(e) => e.key === 'Enter' && joinWaitlist()}
                      className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      disabled={waitlistLoading}
                    />
                    <button 
                      onClick={joinWaitlist}
                      disabled={waitlistLoading}
                      className="btn-primary px-6 disabled:opacity-50"
                    >
                      {waitlistLoading ? 'Joining...' : 'Get Early Access'}
                    </button>
                  </div>
                  {waitlistError && (
                    <p className="text-red-500 text-sm mt-2">{waitlistError}</p>
                  )}
                  <p className="text-slate-500 text-sm mt-3">No spam. Unsubscribe anytime.</p>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-slate-900 text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-4">Ready to Build with Private AI?</h2>
          <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
            Try our free scam detector demo today. 
            Build security agents, fraud detectors, or any intelligent system - 100% locally.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="#scam-detector" className="btn-primary inline-flex items-center gap-2">
              Try Free Demo <ArrowRight className="w-5 h-5" />
            </a>
            <a href="#use-cases" className="btn-secondary">
              See Use Cases
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-slate-950 text-slate-400">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <h4 className="text-white font-bold text-lg mb-4">NomadaLLM</h4>
              <p className="text-sm">
                Embedded Intelligence SDK. Add a brain to any application.
              </p>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#features" className="hover:text-white">Features</a></li>
                <li><a href="#pricing" className="hover:text-white">Pricing</a></li>
                <li><a href="#docs" className="hover:text-white">Documentation</a></li>
                <li><a href="#case-studies" className="hover:text-white">Case Studies</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="https://nomadahealth.com" className="hover:text-white">Nomada Health</a></li>
                <li><a href="/blog" className="hover:text-white">Blog</a></li>
                <li><a href="/careers" className="hover:text-white">Careers</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="/privacy" className="hover:text-white">Privacy Policy</a></li>
                <li><a href="/terms" className="hover:text-white">Terms of Service</a></li>
                <li><a href="/license" className="hover:text-white">License</a></li>
              </ul>
            </div>
          </div>
          
          {/* Tagline */}
          <div className="border-t border-slate-800 pt-8 mb-8">
            <div className="text-center max-w-2xl mx-auto">
              <p className="text-primary-400 text-sm font-medium">
                NomadaLLM: Private AI for developers.
              </p>
              <p className="text-slate-500 text-xs mt-2">
                Your data stays yours. Always.
              </p>
            </div>
          </div>
          
          <div className="border-t border-slate-800 pt-8 text-center text-sm">
            <p className="mb-2">&copy; {new Date().getFullYear()} Nomada Health. All rights reserved.</p>
            <p className="text-slate-500">
              <a href="https://nomadahealth.com" className="hover:text-primary-400">
                Built by Nomada Health
              </a>
            </p>
          </div>
        </div>
      </footer>
    </main>
  )
}
