<script>
	// Kiln admin: operator control surface (SPEC §12 Phase 10, deliberately
	// minimal). Same-origin /admin API; auth is the admin bearer token. SSE
	// is consumed with a streaming fetch because EventSource cannot send an
	// Authorization header.
	let token = $state(
		typeof localStorage === 'undefined' ? '' : (localStorage.getItem('kiln-admin-token') ?? '')
	);
	let connected = $state(false);
	let banner = $state(''); // verbatim API error message, when any
	let models = $state([]);
	let memory = $state(null);
	let jobs = $state([]);
	let jobsError = $state('');
	let download = $state({ repo: '', revision: '', dest: '' });
	let quantize = $state({ path: '', bits: '4', group_size: '64', out: '' });
	let abort = null;
	let jobsTimer = null;

	function authHeaders() {
		return { Authorization: `Bearer ${token}` };
	}

	// Surfaces the API's own error message verbatim — the 403 for a
	// missing admin_token_hash already names the exact fix.
	async function apiError(response) {
		try {
			const body = await response.json();
			if (body?.error?.message) return body.error.message;
		} catch {
			/* non-JSON body */
		}
		return `request failed with HTTP ${response.status}`;
	}

	async function connect() {
		banner = '';
		localStorage.setItem('kiln-admin-token', token);
		const response = await fetch('/admin/models', { headers: authHeaders() }).catch(() => null);
		if (!response) {
			banner = 'gateway unreachable';
			return;
		}
		if (!response.ok) {
			banner = await apiError(response);
			return;
		}
		const body = await response.json();
		models = body.models;
		memory = body.memory;
		connected = true;
		streamStats();
		refreshJobs();
	}

	function disconnect() {
		connected = false;
		abort?.abort();
		clearTimeout(jobsTimer);
	}

	async function streamStats() {
		abort = new AbortController();
		while (connected) {
			try {
				const response = await fetch('/admin/stats', {
					headers: authHeaders(),
					signal: abort.signal
				});
				if (!response.ok) {
					banner = await apiError(response);
					connected = false;
					return;
				}
				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let buffer = '';
				for (;;) {
					const { done, value } = await reader.read();
					if (done) break;
					buffer += decoder.decode(value, { stream: true });
					let end;
					while ((end = buffer.indexOf('\n\n')) >= 0) {
						const frame = buffer.slice(0, end);
						buffer = buffer.slice(end + 2);
						const data = frame
							.split('\n')
							.filter((line) => line.startsWith('data:'))
							.map((line) => line.slice(5).trim())
							.join('');
						if (!data) continue; // keep-alive comment frames
						const snapshot = JSON.parse(data);
						models = snapshot.models;
						memory = snapshot.memory;
					}
				}
			} catch {
				if (!connected) return;
			}
			// Stream dropped (gateway restart?): retry after a beat.
			await new Promise((resolve) => setTimeout(resolve, 1000));
		}
	}

	async function modelAction(id, action, body) {
		banner = '';
		const response = await fetch(`/admin/models/${encodeURIComponent(id)}/${action}`, {
			method: 'POST',
			headers: body
				? { ...authHeaders(), 'content-type': 'application/json' }
				: authHeaders(),
			body: body ? JSON.stringify(body) : undefined
		}).catch(() => null);
		if (!response) {
			banner = 'gateway unreachable';
		} else if (!response.ok) {
			banner = await apiError(response);
		}
	}

	async function refreshJobs() {
		clearTimeout(jobsTimer);
		const response = await fetch('/admin/jobs', { headers: authHeaders() }).catch(() => null);
		if (!response) {
			jobsError = 'gateway unreachable';
		} else if (!response.ok) {
			jobsError = await apiError(response);
		} else {
			jobsError = '';
			jobs = (await response.json()).jobs;
		}
		// Keep polling while anything is still moving.
		if (connected && jobs.some((job) => job.state === 'queued' || job.state === 'running')) {
			jobsTimer = setTimeout(refreshJobs, 1000);
		}
	}

	async function submitJob(kind, payload) {
		jobsError = '';
		const response = await fetch(`/admin/jobs/${kind}`, {
			method: 'POST',
			headers: { ...authHeaders(), 'content-type': 'application/json' },
			body: JSON.stringify(payload)
		}).catch(() => null);
		if (!response) {
			jobsError = 'gateway unreachable';
		} else if (!response.ok) {
			jobsError = await apiError(response);
		} else {
			refreshJobs();
		}
	}

	function submitDownload() {
		const payload = { repo: download.repo };
		if (download.revision) payload.revision = download.revision;
		if (download.dest) payload.dest = download.dest;
		submitJob('download', payload);
	}

	function submitQuantize() {
		const payload = { path: quantize.path };
		if (quantize.bits) payload.bits = Number(quantize.bits);
		if (quantize.group_size) payload.group_size = Number(quantize.group_size);
		if (quantize.out) payload.out = quantize.out;
		submitJob('quantize', payload);
	}

	function mib(bytes) {
		return bytes ? `${(bytes / (1024 * 1024)).toFixed(0)} MiB` : '0';
	}

	function jobDetail(job) {
		const detail = job.detail;
		if (detail == null) return '';
		return typeof detail === 'string' ? detail : JSON.stringify(detail);
	}
</script>

<main>
	<h1>Kiln admin</h1>

	<form
		class="token"
		onsubmit={(event) => {
			event.preventDefault();
			connect();
		}}
	>
		<input
			type="password"
			placeholder="admin token"
			data-testid="token-input"
			bind:value={token}
		/>
		<button type="submit" data-testid="connect">connect</button>
		{#if connected}<span class="ok" data-testid="connected">connected</span>{/if}
	</form>

	{#if banner}
		<p class="banner" data-testid="banner">{banner}</p>
	{/if}

	{#if connected}
		{#if memory}
			<section>
				<h2>memory</h2>
				<p data-testid="memory">
					used {mib(memory.used_bytes)} (+{mib(memory.reserved_bytes)} reserved) of
					{mib(memory.budget_bytes)} budget
				</p>
			</section>
		{/if}

		<section>
			<h2>models</h2>
			<table>
				<thead>
					<tr>
						<th>id</th><th>worker</th><th>status</th><th>pinned</th><th>memory</th>
						<th>reqs</th><th>tokens out</th><th>actions</th>
					</tr>
				</thead>
				<tbody>
					{#each models as model (model.id)}
						<tr data-testid="model-{model.id}">
							<td>{model.id}</td>
							<td>{model.worker}</td>
							<td data-testid="status-{model.id}">{model.status}</td>
							<td data-testid="pinned-{model.id}">{model.pinned ? 'yes' : 'no'}</td>
							<td>{mib(model.usage_bytes)}</td>
							<td data-testid="reqs-{model.id}">
								{model.health ? model.health.requests_running : '–'}
							</td>
							<td data-testid="tokens-{model.id}">
								{model.stats ? model.stats.tokens_generated_total : '–'}
							</td>
							<td>
								<button
									data-testid="load-{model.id}"
									disabled={!model.status.startsWith('unloaded')}
									onclick={() => modelAction(model.id, 'load')}
								>
									load
								</button>
								<button
									data-testid="unload-{model.id}"
									disabled={model.status !== 'ready'}
									onclick={() => modelAction(model.id, 'unload')}
								>
									unload
								</button>
								<button
									data-testid="pin-{model.id}"
									onclick={() => modelAction(model.id, 'pin', { pinned: !model.pinned })}
								>
									{model.pinned ? 'unpin' : 'pin'}
								</button>
							</td>
						</tr>
					{:else}
						<tr><td colspan="8">no models configured</td></tr>
					{/each}
				</tbody>
			</table>
		</section>

		<section>
			<h2>jobs</h2>
			{#if jobsError}
				<p class="banner" data-testid="jobs-error">{jobsError}</p>
			{/if}
			<form
				class="job"
				onsubmit={(event) => {
					event.preventDefault();
					submitDownload();
				}}
			>
				<input placeholder="hf repo (org/name)" data-testid="dl-repo" bind:value={download.repo} />
				<input placeholder="revision (optional)" bind:value={download.revision} />
				<input placeholder="dest (optional)" data-testid="dl-dest" bind:value={download.dest} />
				<button type="submit" data-testid="dl-submit" disabled={!download.repo}>download</button>
			</form>
			<form
				class="job"
				onsubmit={(event) => {
					event.preventDefault();
					submitQuantize();
				}}
			>
				<input placeholder="model path" data-testid="q-path" bind:value={quantize.path} />
				<input placeholder="bits" size="4" bind:value={quantize.bits} />
				<input placeholder="group size" size="6" bind:value={quantize.group_size} />
				<input placeholder="out (optional)" bind:value={quantize.out} />
				<button type="submit" data-testid="q-submit" disabled={!quantize.path}>quantize</button>
			</form>
			<table>
				<thead>
					<tr><th>id</th><th>kind</th><th>state</th><th>detail</th></tr>
				</thead>
				<tbody>
					{#each jobs as job (job.id)}
						<tr data-testid="job-{job.id}">
							<td>{job.id.slice(0, 8)}</td>
							<td>{job.kind}</td>
							<td data-testid="job-state-{job.id}">{job.state}</td>
							<td class="detail">{jobDetail(job)}</td>
						</tr>
					{:else}
						<tr><td colspan="4">no jobs</td></tr>
					{/each}
				</tbody>
			</table>
			<button onclick={refreshJobs} data-testid="jobs-refresh">refresh jobs</button>
		</section>
	{/if}
</main>

<style>
	main {
		font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
		max-width: 64rem;
		margin: 1rem auto;
		padding: 0 1rem;
	}
	h1 {
		font-size: 1.2rem;
	}
	h2 {
		font-size: 1rem;
		margin: 1.5rem 0 0.5rem;
	}
	table {
		border-collapse: collapse;
		width: 100%;
	}
	th,
	td {
		border: 1px solid #ccc;
		padding: 0.25rem 0.5rem;
		text-align: left;
	}
	.banner {
		background: #fee;
		border: 1px solid #c33;
		color: #900;
		padding: 0.5rem;
	}
	.ok {
		color: #080;
	}
	.token,
	.job {
		display: flex;
		gap: 0.5rem;
		margin: 0.5rem 0;
	}
	.token input,
	.job input {
		flex: 1;
	}
	.detail {
		max-width: 24rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	button {
		cursor: pointer;
	}
</style>
