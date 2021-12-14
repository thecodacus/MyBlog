/**
 * Configure your Gatsby site with this file.
 *
 * See: https://www.gatsbyjs.com/docs/gatsby-config/
 */

module.exports = {
	/* Your site config here */
	plugins: [
		"gatsby-plugin-image",
		"gatsby-plugin-sharp",
		"gatsby-transformer-sharp",
		`gatsby-plugin-sass`,
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
							showLineNumbers: false,
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
				publicPath: `/admin`,
				modulePath: `${__dirname}/src/admin/index.js`,
				htmlPath: `${__dirname}/src/admin/index.html`,
			},
		},

		// comment system
		{
			resolve: `gatsby-plugin-disqus`,
			options: {
				shortname: `the-codacus`,
			},
		},
	],
	siteMetadata: {
		url: "thecodacus.com",
		title: "Codacus",
		subHeader: "The Coding Abacus",
		description: "A Decentralized Blog, powered by Gatsby and IPFS",
		copyright: "Copyright 2021 The Codacus",
	},
}
