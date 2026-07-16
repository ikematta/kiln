import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
export default {
	kit: {
		// Static build embedded into kiln-gateway via rust-embed and served
		// at /ui (SPEC §3). Single prerendered page, no fallback router.
		adapter: adapter({ pages: 'build', assets: 'build', fallback: undefined }),
		paths: { base: '/ui' }
	}
};
