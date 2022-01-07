/**
 * Configure your Gatsby site with this file.
 *
 * See: https://www.gatsbyjs.com/docs/gatsby-config/
 */

module.exports = {
	siteMetadata: {
		url: "https://codacus.com",
		siteUrl: "https://codacus.com",
		title: "Codacus",
		subHeader: "The Coding Abacus",
		description: "A Decentralized Blog",
		ps: "- powered by Gatsby and IPFS",
		copyright: "Copyright 2021 The Codacus",
		social: {
			twitter: "thecodacus",
		},
	},
	/* Your site config here */
	plugins: [
		"gatsby-plugin-image",
		"gatsby-plugin-sharp",
		"gatsby-transformer-sharp",
		`gatsby-plugin-sass`,
		"gatsby-plugin-react-helmet",
		// pwa
		{
			resolve: `gatsby-plugin-manifest`,
			options: {
				name: `The Coding Abacus`,
				short_name: `Codacus`,
				start_url: `/`,
				background_color: `#f7f0eb`,
				theme_color: `#a2466c`,
				display: `standalone`,
				icon: `src/images/codacus.png`,
			},
		},
		{
			resolve: `gatsby-plugin-offline`,
		},
		// file systems
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `images`,
				path: `${__dirname}/src/images/`,
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				path: `${__dirname}/static/images`,
				name: "media",
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `posts`,
				path: `${__dirname}/src/posts/`,
			},
		},

		// content parsers
		// Including in your Gatsby plugins will transform any paths in your frontmatter

		{
			resolve: "gatsby-transformer-remark",
			options: {
				plugins: [
					`gatsby-plugin-netlify-cms-paths`,
					{
						resolve: `gatsby-remark-prismjs`,
						options: {
							classPrefix: "language-",
							inlineCodeMarker: null,
							aliases: {},
							showLineNumbers: true,
							noInlineHighlight: false,
							languageExtensions: [
								{
									language: "superscript",
									extend: "javascript",
									definition: {
										superscript_types: /(SuperType)/,
									},
									insertBefore: {
										function: {
											superscript_keywords: /(superif|superelse)/,
										},
									},
								},
							],
							prompt: {
								user: "root",
								host: "localhost",
								global: false,
							},
							// By default the HTML entities <>&'" are escaped.
							// Add additional HTML escapes by providing a mapping
							// of HTML entities and their escape value IE: { '}': '&#123;' }
							escapeEntities: {},
						},
					},
					// {
					// 	resolve: "gatsby-remark-relative-images",
					// 	options: {
					// 		name: "uploads", // Must match the source name ^
					// 	},
					// },
					{
						resolve: `gatsby-remark-images`,
						options: {
							maxWidth: 590,
						},
					},
				],
			},
		},
		// netlifycms
		`gatsby-plugin-netlify-cms-paths`,
		{
			resolve: "gatsby-plugin-netlify-cms",
			options: {
				modulePath: `${__dirname}/src/admin/index.js`,
			},
		},

		// comment system
		{
			resolve: `gatsby-plugin-disqus`,
			options: {
				shortname: `the-codacus`,
			},
		},

		// sitemap
		{
			resolve: `gatsby-plugin-sitemap`,
			options: {
				output: `/sitemap.xml`,
				exclude: ["/admin/**"],
				query: `
					{
						site {
							siteMetadata {
								siteUrl
							}
						}
						allSitePage {
							edges {
								node {
									path
								}
							}
						}
					}
				`,
				serialize: ({ site, allSitePage }) =>
					allSitePage.edges.map(edge => {
						return {
							url: site.siteMetadata.siteUrl + edge.node.path,
							changefreq: `daily`,
							priority: 0.7,
						}
					}),
			},
		},
	],
}
