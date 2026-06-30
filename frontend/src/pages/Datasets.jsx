import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { datasetsApi } from '../api/client'
import { useAppStore } from '../store/appStore'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import { motion } from 'framer-motion'
import {
  Search, Download, Upload, Trash2, Eye, Database,
  CheckCircle, Loader2, FolderOpen, Satellite, CheckCircle2,
  FileText, FileArchive, X, Zap
} from 'lucide-react'
import clsx from 'clsx'

function SearchPanel({ onSelect }) {
  const [ticId, setTicId] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [downloading, setDownloading] = useState(null)
  const qc = useQueryClient()

  const handleSearch = async () => {
    if (!ticId.trim()) return
    setSearching(true)
    try {
      const res = await datasetsApi.search(ticId.trim())
      setSearchResults(res.data)
      if (res.data.length === 0) toast('No TESS observations found for this TIC ID', { icon: '🔭' })
    } catch { /* handled globally */ }
    setSearching(false)
  }

  const handleDownload = async (result) => {
    setDownloading(result.sector)
    const tid = toast.loading(`Downloading TIC ${result.tic_id} S${result.sector}...`)
    try {
      const res = await datasetsApi.download(result.tic_id, result.sector)
      toast.success(`Downloaded: ${res.data.name}`, { id: tid })
      qc.invalidateQueries({ queryKey: ['datasets'] })
      onSelect(res.data)
    } catch { toast.error('Download failed', { id: tid }) }
    setDownloading(null)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-cosmic-cyan">
            <Search size={16} />
          </div>
          <input
            id="tic-search"
            className="input w-full pl-11 font-mono tracking-widest text-lg h-14"
            placeholder="ENTER TIC ID (e.g. 261136679)"
            value={ticId}
            onChange={e => setTicId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
          />
        </div>
        <button id="search-btn" onClick={handleSearch} disabled={searching} className="btn-primary h-14 px-8 text-base">
          {searching ? <Loader2 size={18} className="animate-spin" /> : 'INITIALIZE SCAN'}
        </button>
      </div>
      
      {searchResults.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="space-y-3 mt-6"
        >
          <div className="text-xs text-cosmic-cyan font-display uppercase tracking-widest mb-2 border-b border-white/10 pb-2">Scan Results</div>
          {searchResults.map((r, i) => (
            <motion.div 
              initial={{ opacity: 0, x: -10 }} 
              animate={{ opacity: 1, x: 0 }} 
              transition={{ delay: i * 0.1 }}
              key={i} 
              className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-cosmic-cyan/30 transition-all group"
            >
              <div>
                <div className="font-display font-semibold text-white text-base group-hover:text-cosmic-cyan transition-colors">Sector {r.sector}</div>
                <div className="text-xs text-gray-400 font-sans mt-1">{r.description} {r.exptime ? `| ${r.exptime}s cadence` : ''}</div>
              </div>
              <button
                onClick={() => handleDownload(r)}
                disabled={!!downloading}
                className="btn-outline text-xs px-4 border-cosmic-cyan/30 hover:bg-cosmic-cyan/20"
              >
                {downloading === r.sector ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                ACQUIRE
              </button>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  )
}

function FileIcon({ filename }) {
  if (!filename) return <FileText size={32} className="text-gray-500" />
  const lower = filename.toLowerCase()
  if (lower.endsWith('.csv.gz') || lower.endsWith('.gz')) {
    return <FileArchive size={32} className="text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.5)]" />
  }
  if (lower.endsWith('.csv')) {
    return <FileText size={32} className="text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
  }
  return <FileText size={32} className="text-cosmic-cyan" />
}

function FileTypeBadge({ filename }) {
  if (!filename) return null
  const lower = filename.toLowerCase()
  if (lower.endsWith('.csv.gz')) return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-widest bg-amber-400/10 text-amber-400 border border-amber-400/20">
      <Zap size={10} /> CSV.GZ · Compressed
    </span>
  )
  if (lower.endsWith('.csv')) return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-widest bg-emerald-400/10 text-emerald-400 border border-emerald-400/20">
      <FileText size={10} /> CSV · Plain Text
    </span>
  )
  return null
}

function UploadPanel({ onUpload }) {
  const [name, setName] = useState('')
  const [ticId, setTicId] = useState('')
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const qc = useQueryClient()

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/fits':   ['.fits', '.fit'],
      'text/csv':           ['.csv'],
      'application/gzip':   ['.gz'],
      'application/x-gzip': ['.csv.gz'],
    },
    maxFiles: 1,
    onDrop: (accepted) => {
      if (!accepted.length) return
      setFile(accepted[0])
      if (!name) setName(accepted[0].name.replace(/\.csv\.gz$|\.csv$|\.fits?$/i, ''))
    },
  })

  const clearFile = (e) => { e.stopPropagation(); setFile(null); setName(''); setTicId('') }

  const handleUpload = async () => {
    if (!file || !name.trim()) { toast.error('File and name are required'); return }
    setUploading(true)
    const tid = toast.loading('Transmitting to secure server...')
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name)
    if (ticId) fd.append('tic_id', ticId)
    try {
      const res = await datasetsApi.upload(fd)
      toast.success('Transmission successful!', { id: tid })
      qc.invalidateQueries({ queryKey: ['datasets'] })
      onUpload(res.data)
      setFile(null); setName(''); setTicId('')
    } catch { toast.error('Transmission failed', { id: tid }) }
    setUploading(false)
  }

  return (
    <div className="space-y-5">

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={clsx(
          'relative border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden group',
          isDragActive
            ? 'border-cosmic-cyan bg-cosmic-cyan/10 scale-[0.99]'
            : file
            ? 'border-emerald-400/40 bg-emerald-400/5 hover:border-emerald-400/60'
            : 'border-white/20 hover:border-cosmic-cyan/50 hover:bg-white/[0.03]'
        )}
      >
        <input {...getInputProps()} />

        {/* Animated glow when drag active */}
        {isDragActive && (
          <div className="absolute inset-0 bg-cosmic-cyan/5 blur-3xl pointer-events-none" />
        )}

        {file ? (
          /* File selected state */
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-4 p-5"
          >
            <div className="flex-shrink-0 w-14 h-14 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
              <FileIcon filename={file.name} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-display font-semibold text-white text-sm truncate mb-1.5" title={file.name}>
                {file.name}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <FileTypeBadge filename={file.name} />
                <span className="text-[10px] text-gray-500 font-mono">
                  {file.size < 1024 * 1024
                    ? `${(file.size / 1024).toFixed(1)} KB`
                    : `${(file.size / 1024 / 1024).toFixed(2)} MB`}
                </span>
              </div>
            </div>
            <button
              onClick={clearFile}
              className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-all"
              title="Remove file"
            >
              <X size={16} />
            </button>
          </motion.div>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center justify-center py-10 px-6 text-center">
            <motion.div
              animate={isDragActive ? { scale: 1.15, rotate: 5 } : { scale: 1, rotate: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 18 }}
              className="mb-4"
            >
              <Upload
                size={36}
                className={clsx(
                  'transition-colors',
                  isDragActive ? 'text-cosmic-cyan drop-shadow-[0_0_12px_rgba(34,211,238,0.8)]' : 'text-gray-500 group-hover:text-gray-400'
                )}
              />
            </motion.div>
            <p className="text-sm text-gray-300 font-display mb-2">
              {isDragActive ? 'Release to load file' : 'Drop a file here, or click to browse'}
            </p>
            <div className="flex items-center gap-2 flex-wrap justify-center">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-white/5 text-gray-500 border border-white/10">.CSV</span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-amber-400/10 text-amber-400 border border-amber-400/20">.CSV.GZ</span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-white/5 text-gray-500 border border-white/10">.FITS</span>
            </div>
          </div>
        )}
      </div>

      {/* Metadata inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="relative">
          <label className="absolute -top-2 left-3 text-[10px] text-gray-500 font-mono uppercase tracking-widest bg-space-900 px-1">Dataset Name *</label>
          <input
            className="input h-12 pt-1 w-full"
            placeholder="e.g. TIC 261136679 S14"
            value={name}
            onChange={e => setName(e.target.value)}
          />
        </div>
        <div className="relative">
          <label className="absolute -top-2 left-3 text-[10px] text-gray-500 font-mono uppercase tracking-widest bg-space-900 px-1">TIC ID (optional)</label>
          <input
            className="input h-12 pt-1 w-full"
            placeholder="e.g. 261136679"
            value={ticId}
            onChange={e => setTicId(e.target.value)}
          />
        </div>
      </div>

      {/* Upload button */}
      <motion.button
        whileHover={!uploading && file ? { scale: 1.01 } : {}}
        whileTap={!uploading && file ? { scale: 0.98 } : {}}
        onClick={handleUpload}
        disabled={!file || uploading || !name.trim()}
        className={clsx(
          'btn-primary w-full justify-center h-14 text-base relative overflow-hidden transition-all',
          (!file || !name.trim()) && 'opacity-40 cursor-not-allowed'
        )}
      >
        {uploading ? (
          <><Loader2 size={18} className="animate-spin" /> Transmitting...</>
        ) : (
          <><Upload size={18} /> TRANSMIT DATASET</>
        )}
        {uploading && (
          <motion.div
            className="absolute bottom-0 left-0 h-0.5 bg-cosmic-cyan/70"
            initial={{ width: '0%' }}
            animate={{ width: '100%' }}
            transition={{ duration: 3, ease: 'linear' }}
          />
        )}
      </motion.button>
    </div>
  )
}

export default function Datasets() {
  const [tab, setTab] = useState('search')
  const { activeDataset, setActiveDataset } = useAppStore()
  const qc = useQueryClient()

  const { data: datasets = [], isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list().then(r => r.data),
  })

  const deleteMutation = useMutation({
    mutationFn: (id) => datasetsApi.delete(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['datasets'] }); toast.success('Dataset purged from archives') },
  })

  return (
    <div className="space-y-8 pb-10">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left: Search / Upload */}
        <div className="lg:col-span-8 card flex flex-col h-full min-h-[500px]">
          <div className="flex gap-2 mb-8 bg-space-800/80 p-1.5 rounded-xl border border-white/5 backdrop-blur-md">
            {['search', 'upload'].map(t => (
              <button key={t} onClick={() => setTab(t)} className={clsx(
                'flex-1 py-3 rounded-lg text-xs font-display font-bold uppercase tracking-widest transition-all duration-300 relative overflow-hidden',
                tab === t ? 'text-cosmic-cyan' : 'text-gray-500 hover:text-white hover:bg-white/5'
              )}>
                {tab === t && <motion.div layoutId="tab-bg" className="absolute inset-0 bg-cosmic-cyan/10 border border-cosmic-cyan/30 rounded-lg shadow-glow-cyan" />}
                <span className="relative z-10 flex items-center justify-center gap-2">
                  {t === 'search' ? <Satellite size={16} /> : <Upload size={16} />}
                  {t === 'search' ? 'MAST Archive Link' : 'Local Terminal'}
                </span>
              </button>
            ))}
          </div>

          <motion.div
            key={tab}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="flex-1"
          >
            {tab === 'search'
              ? <SearchPanel onSelect={setActiveDataset} />
              : <UploadPanel onUpload={setActiveDataset} />
            }
          </motion.div>
        </div>

        {/* Right: Active dataset preview */}
        <div className="lg:col-span-4 card flex flex-col">
          <h3 className="section-title flex items-center gap-3 text-lg"><Eye size={18} className="text-cosmic-cyan" />Active Module</h3>
          <p className="section-subtitle text-xs mb-6">Currently loaded target in memory</p>
          
          {activeDataset ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-6 flex-1 flex flex-col"
            >
              <div className="p-5 rounded-2xl bg-cosmic-cyan/5 border border-cosmic-cyan/20 flex items-start gap-4 relative overflow-hidden">
                <div className="absolute -right-4 -top-4 w-16 h-16 bg-cosmic-cyan/20 blur-2xl rounded-full"></div>
                <CheckCircle2 size={24} className="text-cosmic-cyan flex-shrink-0 drop-shadow-glow-cyan" />
                <div>
                  <div className="font-display font-bold text-white text-lg tracking-wide">{activeDataset.name}</div>
                  <div className="text-xs text-cosmic-cyan mt-1 uppercase tracking-widest font-mono">System Ready</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm flex-1 content-start">
                {[
                  ['Identifier', activeDataset.tic_id ? `TIC ${activeDataset.tic_id}` : '—'],
                  ['Source', activeDataset.source?.toUpperCase()],
                  ['File Type', activeDataset.file_type?.toUpperCase() || '—'],
                  ['Sector', activeDataset.sector || '—'],
                  ['Data Points', activeDataset.num_points?.toLocaleString() || '—'],
                  ['Time Span', activeDataset.time_start != null ? `${(activeDataset.time_end - activeDataset.time_start).toFixed(2)} d` : '—'],
                ].map(([k, v], idx) => (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    key={k} 
                    className="p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors"
                  >
                    <div className="text-[10px] text-gray-500 font-sans uppercase tracking-widest mb-1.5">{k}</div>
                    <div className="font-display font-medium text-white">{v}</div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center flex-1 border-2 border-dashed border-white/5 rounded-2xl bg-white/5">
              <FolderOpen size={48} className="text-white/10 mb-4" />
              <p className="text-xs text-gray-400 font-sans max-w-[150px]">Link an archive or upload a dataset to activate</p>
            </div>
          )}
        </div>
      </div>

      {/* All datasets table */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="section-title mb-0 flex items-center gap-3"><Database size={20} className="text-cosmic-purple" />Archive Logs <span className="text-sm font-normal text-gray-500 bg-white/5 px-3 py-1 rounded-full border border-white/10">{datasets.length} Total</span></h3>
        </div>
        
        {isLoading ? (
          <div className="space-y-3">{[...Array(4)].map((_, i) => <div key={i} className="skeleton h-16 rounded-2xl" />)}</div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-16 text-gray-500 text-sm font-sans border-2 border-dashed border-white/5 rounded-2xl bg-white/5">Archive empty. Search MAST or upload a file.</div>
        ) : (
          <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/5">
            <table className="data-table">
              <thead>
                <tr><th className="rounded-tl-xl">Target Name</th><th>Identifier</th><th>Source Array</th><th>Sector</th><th>Data Points</th><th>Acquired On</th><th className="rounded-tr-xl">Actions</th></tr>
              </thead>
              <tbody>
                {datasets.map(d => {
                  const isActive = activeDataset?.id === d.id;
                  return (
                  <tr key={d.id} className={clsx("group transition-colors", isActive ? "bg-cosmic-cyan/10 hover:bg-cosmic-cyan/20" : "hover:bg-white/10")}>
                    <td className="w-1/4">
                      <button onClick={() => setActiveDataset(d)} className={clsx('font-display font-medium text-left flex items-center gap-2 transition-colors', isActive ? 'text-cosmic-cyan' : 'text-gray-200 group-hover:text-white')}>
                        {isActive && <CheckCircle2 size={16} className="text-cosmic-cyan drop-shadow-glow-cyan" />}
                        {d.name}
                      </button>
                    </td>
                    <td className="font-mono text-xs text-gray-400">{d.tic_id || '—'}</td>
                    <td><span className={clsx('badge', d.source === 'mast' ? 'badge-tls' : 'badge-bls')}>{d.source}</span></td>
                    <td className="text-gray-300">{d.sector || '—'}</td>
                    <td className="font-mono text-xs text-gray-400">{d.num_points?.toLocaleString() || '—'}</td>
                    <td className="text-xs text-gray-500">{new Date(d.created_at).toLocaleDateString()}</td>
                    <td>
                      <div className="flex gap-2">
                        {!isActive && (
                          <button onClick={() => setActiveDataset(d)} className="btn-outline px-3 py-1.5 text-xs">Activate</button>
                        )}
                        <button onClick={() => deleteMutation.mutate(d.id)} className="btn-danger px-3 py-1.5 text-xs bg-transparent border-transparent hover:border-cosmic-red/50">
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )})}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>
    </div>
  )
}
